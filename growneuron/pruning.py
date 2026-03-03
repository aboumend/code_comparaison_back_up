import tensorflow as tf
import numpy as np

class Masking(object):
    def __init__(self):
        self.masks = {}
        self.modules = []
        self.names = []

    def init(self, mode='ERK', density=0.5, erk_power_scale=1.0):
        self.density=density
        if (mode=='magnitude'):
            print("applying magnitude pruning")
            weight_abs=[]
            for i in range(len(self.module.layers)):
                name=getattr(self.module.layers[i], 'name')
                if name not in self.masks: #on exclue les layers de batch norm, pooling et flatten car on ne les elague pas
                    continue
                weights=self.module.layers[i].get_weights()[0]
                weight_abs.append(abs(weights)) #on mets toutes les matrices de poids en valeurs absolue dans le weight_abs 
            
            all_scores=tf.concat([tf.reshape(x, [-1]) for x in weight_abs], axis=0) #on fait un flatten (avec le tf.reshape) et on concatene pour avoir 
                                                                                    #tous les poids dans un tensor en 1 dim
            print("all scores shape/ nb prunable params", all_scores.shape, "=?", len(all_scores))
            num_params_to_keep = int(len(all_scores) * self.density) #on calcule le nb de param a garder
            print("num parameters to keep", num_params_to_keep)
            res=tf.math.top_k(all_scores, k=num_params_to_keep) #on prend les k=num_params_to_keep plus grande valeur
            threshold=res.values.numpy()
            print("threshold", threshold)
            acceptable_score = threshold[-1] #on prend le poids le plus petit parmi les k plus grand (k=num_params_to_keep)
                                             #c'est le seuil d'elagage, tous les poids inferieurs a acceptable_score seront elagues
            print("acceptable score", acceptable_score)

            for i in range(len(self.module.layers)):
                name=getattr(self.module.layers[i], 'name')
                if name not in self.masks: #on exclue les layers de batch norm, pooling et flatten car on ne les elague pas
                    continue
                weights=self.module.layers[i].get_weights()[0]
                self.masks[name] = tf.cast(((abs(weights)) >= acceptable_score), float)#on elague (mask=1) si la val absolu du poids est inferieur au seuil definie
           

        elif (mode=='ERK'):
            print("applying random pruning with ERK approach")
            total_params = 0
            for name, weight in self.masks.items():#masks contient des tensor qui contiennent des zero c'est initialise dans la methode add_module
                                                   #chaque element de masque correpond au tensor qui contient les poids du layer donc on a un mask par layer
                total_params += tf.size(weight) #nb d'elements de weight (nb element du layer)
                                               #dans total_param on a le nb de poids du network
                
            print("total_params", total_params)
            is_epsilon_valid = False
            dense_layers=set() #ensemble note entre {}, les elements du set sont non ordonne, inchangeable et on n'a pas de valeur en double

            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}#dictionnaire
                #le but de cette boucle est de trouver les layers pour qui la sparsity doit etre 0
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape) #retourne le produit des elements de shape(donc produit des dim du tensor) pour avoir le nb d'elements/poids du tensor
                    n_zeros = n_param * (1 - self.density) #nb de param a mettre a zero (a elaguer), density c'est le pourcentage de poids a garder
                    n_ones = n_param * self.density #nb de param a mettre a 1 

                    if name in dense_layers:
                        rhs -= n_zeros #on soustrait le nb de param elagues
                    else:
                        rhs += n_ones #on rajoute le nb de param non elagues
                        raw_probabilities[name] = (#je pense pourcentage de poids a elaguer/ raw_probabilities est un dictionnaire qui contient la proba associe a chaque tensor de poids qui a la cle name
                                                          np.sum(mask.shape) / np.prod(mask.shape) #somme des dim du tensor/produit des dim, le produit des dim c'est le nb d'element
                                                  ) ** erk_power_scale #doit etre un param de erk
                        divisor += raw_probabilities[name] * n_param #je pense c'est le nb de poids qu'on garde
                epsilon = rhs / divisor #divisor je pense c'est le nb total de poids qu'on garde et
                                        #rhs c'est un peu un compteur a qui on rajoute le nb de poids non elagues et on deduit le nb de poids elagues
                max_prob = np.max(list(raw_probabilities.values()))#on prend le max des values du dictionnaies
                max_prob_one = max_prob * epsilon#proba d'elagage max *epsilon je pense
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():#items c'est les elem du dictionnaire en (cle, valeur)
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True
            
            density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():
                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
                )
                self.masks[name] = tf.cast(tf.random.uniform(mask.shape, minval=0, maxval=1, dtype=tf.dtypes.float32) < density_dict[name], float)
                
                total_nonzero += density_dict[name] * tf.get_static_value(tf.size(mask))
                
            print(f"Overall sparsity {total_nonzero / total_params}")
        
        self.apply_mask()
        total_size = 0
        sparse_size = 0
        for i in range(len(self.module.layers)):
            name=getattr(self.module.layers[i], 'name')
            if name in self.masks:
                weight=self.module.layers[i].get_weights()[0]
                print(name, 'density:',np.sum((weight!=0))/tf.size(weight))
                total_size+=tf.size(weight)
                sparse_size+=np.sum((weight!=0))
            
        print('Total Model parameters:', total_size)
        print('number of papameters set to 0:', sparse_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(self.density, sparse_size / total_size))



    def add_module(self, module, density, prune_type):
        self.modules.append(module)
        self.module=module
        for i in range(len(self.module.layers)):
            name=getattr(self.module.layers[i], 'name')
            self.names.append(name)
            #print(name)
            if("activation" not in (name)) and ("average_pooling" not in (name)) and ("flatten" not in (name)):
                self.masks[name]=tf.zeros(self.module.layers[i].get_weights()[0].shape ,dtype=tf.dtypes.float32)

        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing batch norms...')
        self.remove_type(tf.keras.layers.BatchNormalization)
        print('Removing activations...')
        self.remove_weight_partial_name('activation')

        print("layers associated with mask/ to be pruned:", list(self.masks.keys()))

        self.init(mode=prune_type, density=density)


    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} '.format(name, self.masks[name].shape))
            self.masks.pop(name)
        
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                                   np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for i in range(len(self.module.layers)):
            if isinstance(self.module.layers[i], nn_type):
                name=getattr(self.module.layers[i], 'name')
                self.remove_weight(name)

    def apply_mask(self):
        for i in range(len(self.module.layers)):
            name=getattr(self.module.layers[i], 'name')
            if name in self.masks:
                #print("applying masks to", name)
                weights=self.module.layers[i].get_weights()[0]
                #print("weights before pruning", self.arch.layers[i].get_weights())
                
                new_weights=tf.math.multiply(weights,self.masks[name])
                new_weights=tf.convert_to_tensor(new_weights)
                new_weights_list=[]
                new_weights_list.append(new_weights)

                #la condition c'est si le layer contient un bias
                #la facon dont c'est represente est que chaque layer est associe a une liste (self.module.layers[i].get_weights())
                #cette liste contient la matrice des poids
                #s'il y a un bias cette liste contient un second element qui est le bias
                #on elague juste les poids donc on applique le masque a la matrice des poids et s'il y a un bias on le garde inchange
                #on concatene la matrice de poids elaguee au bias et on affecte les nouveaux poids au layer
                if (len(self.module.layers[i].get_weights())>1):
                    new_weights_list.append(self.module.layers[i].get_weights()[1])


                self.module.layers[i].set_weights(new_weights_list)
                
                #print("weight after pruning", self.arch.layers[i].get_weights())
                #print("----------------------------------------------------------------------")
