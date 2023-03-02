
# Generate dictionaries of parameters using these:
            # data,
            # train_size=0.8,
            # epochs=300,
            # batch_size=128,
            # lr=0.001,
            # b1=0.9,
            # b2=0.999,
            # clip_value=None,
            # random_state=None
# data is always going to be the same
# always add a random_state for reproducibility


d1 = {
    'train_size': 0.8,
    'epochs': 300,
    'batch_size': 128,
    'lr': 0.001,
    'b1': 0.9,
    'b2': 0.999,
    'clip_value': None,
    'random_state': 0
}

d2 = {
    'train_size': 0.8,
    'epochs': 200,
    'batch_size': 1024,
    'lr': 0.001,
    'b1': 0.9,
    'b2': 0.999,
    'clip_value': None,
    'random_state': 1
}

d3 = {
    'train_size': 0.8,
    'epochs': 300,
    'batch_size': 128,
    'lr': 0.001,
    'b1': 0.9,
    'b2': 0.999,
    'clip_value': 0.01,
    'random_state': 2
}

d4 = {
    'train_size': 0.8,
    'epochs': 200,
    'batch_size': 1024,
    'lr': 0.001,
    'b1': 0.9,
    'b2': 0.999,
    'clip_value': 0.01,
    'random_state': 3
}

d5 = {
    'train_size': 0.95,
    'epochs': 500,
    'batch_size': 256,
    'lr': 0.005,
    'b1': 0.9,
    'b2': 0.99,
    'clip_value': 0.1,
    'random_state': 4
}

