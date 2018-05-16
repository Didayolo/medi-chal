from keras.models import Sequential
from keras.layers import Dense, Dropout

def get_NN(input_dim):
	model = Sequential()
	model.add(Dense(12, input_dim=input_dim, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(1, activation='sigmoid'))
	return model


def training(X_train, y_train, X_test, y_test, verbose=1):
	model = get_NN(X_train.shape[1])

	# Compile model.
	model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Fit the model.
	history = model.fit(X_train, y_train, epochs=10, batch_size=12, validation_data=(X_test, y_test), verbose=verbose)

	return history