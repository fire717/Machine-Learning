from sklearn.utils import class_weight


class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(label_list),
                                                  label_list)
                                                  
                                                  
                                                  
model.fit_generator(...,
                    class_weight={'outputs':class_weights},
                    )
#'outputs' is the output (which u want to balance) layer name 
