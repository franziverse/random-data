import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# Function to visualize data samples
def visualize_data(data, labels, typ, horizontal, title, ax=None, fig_height=2, hspace=0.5):
    top_space = 0.7 if horizontal else 0.85
    number_of_pictures = 10 if typ=="MNIST" else 6
    rows = 1 if horizontal else 2
    columns = int(number_of_pictures / rows)
    w = 15 if horizontal else 6
    size = (w, fig_height*rows)

    
    if ax is None:
        fig = plt.figure(figsize=size)
        gs = plt.GridSpec(rows, columns, figure=fig, wspace=0.4, hspace=hspace)
        plt.subplots_adjust(top=top_space)
        fig.suptitle(title, fontsize=16, y=1)
        axs =[fig.add_subplot(gs[a,b]) for a in range(rows) for b in range(columns)]
    else:
        axs = []
        width = 1 / columns
        height = 1 / rows
        axs = [ax.inset_axes([
            col*width + 0.02,
            1-(row+1)*height+0.02,
            width-0.04,
            height-0.1])
            for row in range(rows)
            for col in range(columns)
        ]

    for j in range(number_of_pictures):
        axs[j].set_title(f"Label:{labels[j]}")

        if typ =="MNIST1D":
            axs[j].plot(data[j])
            axs[j].set_xticks([])
            axs[j].set_yticks([])
        elif typ == "MNIST":
            axs[j].imshow(data[j].reshape(28,28), cmap='gray')
            axs[j].axis('off')
        elif typ == "CIFAR10":
            axs[j].imshow(np.array(data[j]).reshape(32,32,3))
            axs[j].axis('off')

    if ax is None:
        plt.show()
    else:
        return axs

# Function to plot training/testing curves
def plot_training(name, training, testing = None, ax = None):
    if ax is None:
        fig, ax = plt.figure(figsize=(8, 4))
    ax.plot(training, 'r-')
    if testing is not None:
        ax.plot(testing, 'b-')
        ax.legend(['Train', 'Test'])
    ax.set_ylabel(name); ax.set_xlabel('Epoch'); ax.grid(True)

    if ax is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax

# Function to plot a list of curves on a single plot
def plot_list(name, ax_name, lst, pos, ax=None):
    if ax is None:
        fig, ax = plt.figure(figsize=(8, 6))

    for i in lst:
        plt.plot(i[pos], label=i[0])
    plt.ylabel(ax_name); plt.xlabel('Epoch'); plt.title(name); plt.grid(True); plt.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax

# Function to create a summary visualization for a single dataset
def summary(title, data_x, data_y, name_dataset, loss_train, acc_train, loss_test, acc_test, fig_height=5, vspace=0.5):
    plt.figure(figsize=(15, fig_height))
    plt.suptitle(title, fontsize=16)
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=vspace)

    ax1 = plt.subplot(grid[0:,0])
    ax2 = plt.subplot(grid[0,1])
    ax3 = plt.subplot(grid[1,1])

    visualize_data(data_x, data_y, name_dataset, False, title, ax=ax1)
    ax1.axis('off')
    plot_training('Loss', loss_train, loss_test, ax = ax2)
    plot_training('Accuracy', acc_train, acc_test, ax = ax3)

# Function to create a summary visualization comparing statistics across datasets
def summary_statistics(losses, accuracies, pos=1, fig_height=5, vspace=0.5):
    plt.figure(figsize=(15, fig_height))
    grid = plt.GridSpec(1,2, wspace=0.3, hspace=vspace)

    ax1 = plt.subplot(grid[0,0])
    for i in losses:
        plt.plot(i[pos], label=f"{i[0]}")
    plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.title('Losses'); plt.grid(True); plt.legend()

    ax2 = plt.subplot(grid[0,1])
    for i in accuracies:
        plt.plot(i[pos], label=f"{i[0]}")
    plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.title('Accuracies'); plt.grid(True); plt.legend()