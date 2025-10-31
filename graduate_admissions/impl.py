from app import Net

if __name__ == '__main__':
    """
    Test different neural network configurations for graduate admissions prediction.
    
    This script evaluates various combinations of:
    - Hidden layer sizes (3, 50, 100, 1000)
    - Activation functions (sigmoid, tanh)
    - Dropout regularization
    """
    
    print("Testing different activation functions and architectures...\n")
    
    # Test 1: Small network with sigmoid activation, no dropout
    print("=" * 50)
    print("Test 1: Hidden size=3, Activation=sigmoid, Dropout=False")
    print("=" * 50)
    net = Net(hidden_size=3, activation_fn='sigmoid', apply_dropout=False)
    result_3_sigmoid = Net.train_and_evaluate_model(net)
    
    # Test 2: Medium network with tanh activation
    print("\n" + "=" * 50)
    print("Test 2: Hidden size=50, Activation=tanh, Dropout=False")
    print("=" * 50)
    net = Net(hidden_size=50, activation_fn='tanh')
    result_50_tanh = Net.train_and_evaluate_model(net)
    
    # Test 3: Large network with tanh activation
    print("\n" + "=" * 50)
    print("Test 3: Hidden size=1000, Activation=tanh, Dropout=False")
    print("=" * 50)
    net = Net(hidden_size=1000, activation_fn='tanh')
    result_1000_tanh = Net.train_and_evaluate_model(net)
    
    # Test 4: Large network with tanh activation and dropout
    print("\n" + "=" * 50)
    print("Test 4: Hidden size=1000, Activation=tanh, Dropout=True")
    print("=" * 50)
    net = Net(hidden_size=1000, activation_fn='tanh', apply_dropout=True)
    result_1000_tanh_dropout = Net.train_and_evaluate_model(net)
    
    # Test 5: Medium network with sigmoid activation
    print("\n" + "=" * 50)
    print("Test 5: Hidden size=100, Activation=sigmoid, Dropout=False")
    print("=" * 50)
    net = Net(hidden_size=100, activation_fn='sigmoid')
    result_100_sigmoid = Net.train_and_evaluate_model(net)
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)