import matplotlib.pyplot as plt

def draw_graph(train_result, valid_result, title, type):
    plt.xlabel('Epoch')
    plt.ylabel(type)
    plt.plot(list(range(len(train_result))), train_result, list(range(len(train_result))), valid_result)
    plt.legend(['Train', 'Validation'])
    plt.savefig(f'graphs/{title}_{type.lower()}.png')
    plt.close()


