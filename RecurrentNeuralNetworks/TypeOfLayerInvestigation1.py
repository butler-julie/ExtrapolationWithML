from RNNSupport import *

def dnn(length_of_sequences, hidden_neurons, loss, optimizer, activation_function, batch_size = None, stateful = False):
    in_out_neurons = 1
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=2, activation=activation_function))
    #model.add(Dense(hidden_neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")  
    return model

def simplernn(length_of_sequences, hidden_neurons, loss, optimizer, activation_function, batch_size = None, stateful = False):
    in_out_neurons = 1
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons)) 
    rnn1 = SimpleRNN(hidden_neurons, 
                    activation=activation_function,
                    return_sequences=False,
                    stateful = stateful,
                    name="RNN1", use_bias=True)(inp)
    dens = Dense(in_out_neurons,name="dense")(rnn1)
    model = Model(inputs=[inp],outputs=[dens])
    model.compile(loss=loss, optimizer=optimizer)  
    return model


def gru(length_of_sequences, hidden_neurons, loss, optimizer, activation_function, batch_size = None, stateful = False):
    in_out_neurons = 1
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons)) 
    rnn1 = GRU(hidden_neurons, 
                    return_sequences=False,
                    activation = activation_function,
                    stateful = stateful,
                    name="RNN1", use_bias=True)(inp)
    dens = Dense(in_out_neurons,name="dense")(rnn1)
    model = Model(inputs=[inp],outputs=[dens])
    model.compile(loss=loss, optimizer=optimizer)  
    return model

def lstm(length_of_sequences, hidden_neurons, loss, optimizer, activation_function, batch_size = None, stateful = False):
    in_out_neurons = 1
    inp = Input(batch_shape=(batch_size, 
                length_of_sequences, 
                in_out_neurons)) 
    rnn1 = LSTM(hidden_neurons, 
                    return_sequences=False,
                    activation = activation_function,
                    stateful = stateful,
                    name="RNN1", use_bias=True)(inp)
    dens = Dense(in_out_neurons,name="dense")(rnn1)
    model = Model(inputs=[inp],outputs=[dens])
    model.compile(loss=loss, optimizer=optimizer)  
    return model

# Vary Dimension
#datatype='VaryDimension'
#X_tot = np.arange(2, 42, 2)
y_tot = np.array([-0.03077640549, -0.08336233266, -0.1446729567, -0.2116753732, -0.2830637392, -0.3581341341, -0.436462435, -0.5177783846,
	-0.6019067271, -0.6887363571, -0.7782028952, -0.8702784034, -0.9649652536, -1.062292565, -1.16231451, 
	-1.265109911, -1.370782966, -1.479465113, -1.591317992, -1.70653767])

dim=12
y_train = y_tot[:dim]

seq = 2
X_train, y_train = format_data_dnn (y_train, seq)

print(X_train)
hidden_neurons = 200
loss = 'mse'
optimizer = 'adam'

activations = [None, 'tanh', 'relu', 'sigmoid', 'softmax']



for act in [None, 'tanh', 'relu', 'sigmoid', 'softmax']:
    errors = []
    for i in range (10):
        model = dnn(seq, hidden_neurons, loss, optimizer, act)
        iterations = 250
        model.fit (X_train, y_train, epochs=iterations, batch_size=10, validation_split=0.0, verbose=True)

        y_return = []

        y_return = y_tot[:dim].tolist()
        next_input = np.array([[[y_return[-2]], [y_return[-1]]]])
        last = [y_return[-1]]

        total_points = 20

        while len(y_return) < total_points:
            next = model.predict(next_input)
            y_return.append(next[0][0])
            next_input = np.array([[last, next[0]]])
            last = next[0]
        mse_err = mse(y_return, y_tot)
        errors.append (mse_err)


    errors = np.asarray(errors)
    print('ACTIVATION FUNCTION: ', act)
    print('AVERAGE MSE ERROR: ', errors.mean())
    print('MAXIMUM MSE ERROR: ', np.amax(errors))
    print('MINIMUM MSE ERROR: ', np.amin(errors))        
    print('******************************')
