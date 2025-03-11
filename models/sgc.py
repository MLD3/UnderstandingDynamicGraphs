import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

class SimpleGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, units=128, activation=tf.nn.relu, dropout=.2, g_dropout=.5, bias=False, k=2): 
        super(SimpleGraphConvolution, self).__init__()
        self.units = units
        self.activation = activation
        self.dropout = dropout
        self.g_dropout = g_dropout
        self.bias = bias
        self.k = k
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_normal', trainable=True)
        if self.bias: 
            self.b = self.add_weight(shape=(1, ), initializer='glorot_normal', trainable=True)
            
    def call(self, inputs, adj): 
        x = inputs
        
        if self.dropout: 
            x = tf.nn.dropout(x, self.dropout)
            
        if self.g_dropout:
            adj = tf.sparse.to_dense(adj)
            adj = tf.sparse.from_dense(tf.nn.dropout(adj, self.g_dropout)) # cast to dense then recast to sparse
        
        for _ in range(self.k): 
            x = tf.sparse.sparse_dense_matmul(adj, x)
            
        x = tf.linalg.matmul(x, self.w)
        
        if self.bias: 
            x += self.b
            
        x = self.activation(x)
            
        return x
    
class SGC(tf.keras.Model): 
    def __init__(self, 
                 num_layers, 
                 n_classes, 
                 bias,
                 dropout,
                 g_dropout,
                 optimizer,
                 loss):
        super(SGC, self).__init__()
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.bias = bias
        self.dropout = dropout
        self.g_dropout = g_dropout
        self.optimizer = optimizer
        self.loss = loss
        self.model_layers = [SimpleGraphConvolution(units=1, bias=bias, dropout=dropout, g_dropout=g_dropout, activation=tf.nn.sigmoid, k=num_layers)]
                      
    def call(self, inputs, adj):
        x = inputs
        for i in range(len(self.layers)): 
            x = self.layers[i](inputs=x, adj=adj)
        
        return x
    
    @tf.function
    def train_step(self, data):
        X, y, y_mask, adj, adj_mean, adj_norm = data
        y_mask = tf.cast(y_mask, tf.bool)
        with tf.GradientTape() as tape:
            y_hat = self(inputs=X, adj=tf.sparse.from_dense(adj_norm), training=True)
            y_hat = tf.boolean_mask(y_hat, y_mask)
            y = tf.boolean_mask(y, y_mask)
            loss_value = self.loss(y, y_hat)
            
        grads = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return y, y_hat
    
    @tf.function
    def test_step(self, data):
        X, y, y_mask, adj, adj_mean, adj_norm = data
        y_mask = tf.cast(y_mask, tf.bool)
        y_hat = self(inputs=X, adj=tf.sparse.from_dense(adj_norm), training=False)
        y_hat = tf.boolean_mask(y_hat, y_mask)
        y = tf.boolean_mask(y, y_mask)
        
        return y, y_hat
        
        
    def train(self, train_dataset, valid_dataset, epochs=10, early_stopping=5, verbose=False):
        early_stopping_count = 0
        best_loss = 100
        train_bce = tf.keras.metrics.BinaryCrossentropy()
        valid_bce = tf.keras.metrics.BinaryCrossentropy()
        
        for epoch in range(epochs): 
            for step, data in enumerate(train_dataset):
                y, y_hat = self.train_step(data)
                train_bce.update_state(y, y_hat)
            
            # early stopping
            for step, data in enumerate(valid_dataset):
                y, y_hat = self.test_step(data)
                valid_bce.update_state(y, y_hat)
            
            loss_value = float(valid_bce.result())
            if loss_value < (best_loss - .01): 
                best_loss = loss_value
                early_stopping_count = 0
            else: 
                early_stopping_count+=1
                    
            if early_stopping_count == early_stopping: 
                if verbose:
                    print(f'Validation loss has not increased for {early_stopping} epochs')
                break
                
            if verbose:
                print(f'Epoch {epoch}: Training loss is {float(train_bce.result()):.3f}, Validation loss is {float(valid_bce.result()):.3f}')
            
            train_bce.reset_states()
            valid_bce.reset_states()
            
    def evaluate(self, valid_dataset, test_dataset, verbose=False): 
        val_auc = tf.keras.metrics.AUC()
        test_auc = tf.keras.metrics.AUC()
        aucs = []
        preds = []
        
        for step, data in enumerate(valid_dataset):
            y, y_hat = self.test_step(data)
            val_auc.update_state(y, y_hat)
        
        for step, data in enumerate(test_dataset):
            y, y_hat = self.test_step(data)
            if tf.unique(y)[0].shape[0] > 1:
                test_auc.update_state(y, y_hat)
                aucs.append(roc_auc_score(y, y_hat))
            else: 
                aucs.append(np.inf)
            preds.append(y_hat)
            
        aucs = np.array(aucs)
        val_auc = float(val_auc.result())
        test_auc = float(test_auc.result())
        test_error = 1.96 * np.std(aucs[np.isinf(aucs) == 0]) / np.sqrt(aucs[np.isinf(aucs) == 0].shape[0])
        
        if verbose:
            print(f'GCN AUC on the test set is {auc:.4f}')
        
        return test_auc, test_error, aucs, val_auc, preds