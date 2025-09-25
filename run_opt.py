import mnist_loader
from network import Network, CrossEntropyCost

# Cargar datos
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Crear la red con CrossEntropyCost
net = Network([784, 30, 10], cost=CrossEntropyCost)
net.default_weight_initializer()

# Probar Adam (10 épocas)
net.SGD_opt(training_data, epochs=10, mini_batch_size=10, eta=0.001,
            test_data=test_data, optimizer='adam',
            opt_params={'beta1':0.9, 'beta2':0.999, 'eps':1e-8})
