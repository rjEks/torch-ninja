from app import Net

if __name__ == '__main__':
    net = Net(hidden_size=3, activation_fn='sigmoid', apply_dropout=False)
    result_3_sigmoid = Net.train_and_evaluate_model(net)

    net = Net(hidden_size=50, activation_fn='tanh')
    result_50_tanh = Net.train_and_evaluate_model(net)

    net = Net(hidden_size=1000, activation_fn='tanh')
    result_1000_tanh = Net.train_and_evaluate_model(net)

    net = Net(hidden_size=1000, activation_fn='tanh', apply_dropout=True)
    result_1000_tanh = Net.train_and_evaluate_model(net)

    net = Net(hidden_size=100, activation_fn='sigmoid')
    result_100_sigmoid = Net.train_and_evaluate_model(net)