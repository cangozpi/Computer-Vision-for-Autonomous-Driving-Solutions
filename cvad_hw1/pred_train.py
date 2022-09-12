import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor

# imports below are added by me
import matplotlib.pyplot as plt


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    # Leverage GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # training config
    regression_loss = torch.nn.L1Loss()
    classification_loss = torch.nn.CrossEntropyLoss()

    # Evaluate in minibatches
    final_cum_loss = 0
    final_loss_lane_dist = 0
    final_loss_route_angle = 0
    final_loss_tl_dist = 0
    final_loss_tl_state = 0
    for img, measurements in dataloader:
        # Extract input features/targets from measurements
        command, _, _, _, _, _, route_angle, lane_dist, _, _, _, tl_state, tl_dist, _ = measurements

        # Leverage GPU if available
        if device != "cpu":
            route_angle = route_angle.cuda()
            lane_dist = lane_dist.cuda()
            tl_state = tl_state.cuda()
            tl_dist = tl_dist.cuda()
            img = img.cuda()
        
        # Forward pass 
        pred_lane_dist, pred_route_angle, pred_tl_dist, pred_tl_state = model(img.float(), command)

        # Calculate Task Block losses individually
        # Regression losses:
        loss_lane_dist = regression_loss(pred_lane_dist, lane_dist)
        loss_route_angle = regression_loss(pred_route_angle ,route_angle)
        loss_tl_dist = regression_loss(pred_tl_dist, tl_dist)
        # Classification loss: 
        loss_tl_state = classification_loss(pred_tl_state, tl_state)

        # Sum Task Block losses
        cum_loss = loss_lane_dist + loss_route_angle + loss_tl_dist + loss_tl_state

        # record the loss
        batch_size = len(img)
        final_loss_lane_dist += (loss_lane_dist.item() * batch_size)
        final_loss_route_angle += (loss_route_angle.item() * batch_size)
        final_loss_tl_dist += (loss_tl_dist.item() * batch_size)
        final_loss_tl_state += (loss_tl_state.item() * batch_size)
        final_cum_loss += (cum_loss.item() * batch_size)

    # calculate mean loss
    final_loss_lane_dist /= len(dataloader.dataset)
    final_loss_route_angle /= len(dataloader.dataset)
    final_loss_tl_dist /= len(dataloader.dataset)
    final_loss_tl_state /= len(dataloader.dataset)
    final_cum_loss /= len(dataloader.dataset)
    
    #print("Average validation loss for the last epoch: ", final_cum_loss)
    return final_cum_loss, final_loss_lane_dist, final_loss_route_angle, final_loss_tl_dist, final_loss_tl_state 
    
    


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    # Leverage GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    
    # training config
    lr = 5e-5
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    regression_loss = torch.nn.L1Loss()
    classification_loss = torch.nn.CrossEntropyLoss()

    # Train in minibatches
    final_cum_loss = 0
    final_loss_lane_dist = 0
    final_loss_route_angle = 0
    final_loss_tl_dist = 0
    final_loss_tl_state = 0
    for i, (img, measurements) in enumerate(dataloader):
        if i % 250 == 0:
            print("Batch: ", i, "/", len(dataloader))
        # Extract input features/targets from measurements
        command, _, _, _, _, _, route_angle, lane_dist, _, _, _, tl_state, tl_dist, _ = measurements

        # Leverage GPU if available
        if device != "cpu":
            route_angle = route_angle.cuda()
            lane_dist = lane_dist.cuda()
            tl_state = tl_state.cuda()
            tl_dist = tl_dist.cuda()
            img = img.cuda()

        # Forward pass 
        pred_lane_dist, pred_route_angle, pred_tl_dist, pred_tl_state = model(img.float(), command)

        # Calculate Task Block losses individually
        # Regression losses:
        loss_lane_dist = regression_loss(pred_lane_dist, lane_dist)
        loss_route_angle = regression_loss(pred_route_angle ,route_angle)
        loss_tl_dist = regression_loss(pred_tl_dist, tl_dist)
        # Classification loss: 
        loss_tl_state = classification_loss(pred_tl_state, tl_state)

        # Sum Task Block losses
        cum_loss = loss_lane_dist + loss_route_angle + loss_tl_dist + loss_tl_state

        # Backpropagate
        optimizer.zero_grad()
        cum_loss.backward()
        optimizer.step()

        # record the loss
        # record total batch losses /convert mean batch losses to cumulative losses
        batch_size = len(img)
        final_loss_lane_dist += (loss_lane_dist.item() * batch_size)
        final_loss_route_angle += (loss_route_angle.item() * batch_size)
        final_loss_tl_dist += (loss_tl_dist.item() * batch_size)
        final_loss_tl_state += (loss_tl_state.item() * batch_size)
        final_cum_loss += (cum_loss.item() * batch_size)

    # calculate averaged loss for the current epoch
    final_loss_lane_dist /= len(dataloader.dataset)
    final_loss_route_angle /= len(dataloader.dataset)
    final_loss_tl_dist /= len(dataloader.dataset)
    final_loss_tl_state /= len(dataloader.dataset)
    final_cum_loss /= len(dataloader.dataset)

    print("Average training loss for the last epoch: ", final_cum_loss, "lane_dist loss: ", final_loss_lane_dist, "route_angle loss: ", final_loss_route_angle \
        , "tl_dist loss: ", final_loss_tl_dist, "tl_state loss: ", final_loss_tl_state)
    return final_cum_loss, final_loss_lane_dist, final_loss_route_angle, final_loss_tl_dist, final_loss_tl_state 
    

def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    train_final_cum_loss, train_final_loss_lane_dist, train_final_loss_route_angle, train_final_loss_tl_dist, train_final_loss_tl_state = [], [], [], [], []
    test_final_cum_loss, test_final_loss_lane_dist, test_final_loss_route_angle, test_final_loss_tl_dist, test_final_loss_tl_state = [], [], [], [], []
    
    for e in train_loss:
        train_final_cum_loss.append(e[0])
        train_final_loss_lane_dist.append(e[1])
        train_final_loss_route_angle.append(e[2])
        train_final_loss_tl_dist.append(e[3])
        train_final_loss_tl_state.append(e[4])

    for e in val_loss:
        test_final_cum_loss.append(e[0])
        test_final_loss_lane_dist.append(e[1])
        test_final_loss_route_angle.append(e[2])
        test_final_loss_tl_dist.append(e[3])
        test_final_loss_tl_state.append(e[4])

        
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.plot(range(len(train_final_cum_loss)), train_final_cum_loss, label="train_final_cum_loss")
    plt.plot(range(len(train_final_loss_lane_dist)), train_final_loss_lane_dist, label="train_final_loss_lane_dist")
    plt.plot(range(len(train_final_loss_route_angle)), train_final_loss_route_angle, label="train_final_loss_route_angle")
    plt.plot(range(len(train_final_loss_tl_dist)), train_final_loss_tl_dist, label="train_final_loss_tl_dist")
    plt.plot(range(len(train_final_loss_tl_state)), train_final_loss_tl_state, label="train_final_loss_tl_state")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.title("Training Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(range(len(test_final_cum_loss)), test_final_cum_loss, label="test_final_cum_loss")
    plt.plot(range(len(test_final_loss_lane_dist)), test_final_loss_lane_dist, label="test_final_loss_lane_dist")
    plt.plot(range(len(test_final_loss_route_angle)), test_final_loss_route_angle, label="test_final_loss_route_angle")
    plt.plot(range(len(test_final_loss_tl_dist)), test_final_loss_tl_dist, label="test_final_loss_tl_dist")
    plt.plot(range(len(test_final_loss_tl_state)), test_final_loss_tl_state, label="test_final_loss_tl_state")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.title("Validation Loss Curve")
    
    plt.savefig("Affordance Prediction Training and Validation Loss Curves")

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(range(len(train_final_loss_lane_dist)), train_final_loss_lane_dist, label="train_final_loss_lane_dist")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,2)
    plt.plot(range(len(train_final_loss_route_angle)), train_final_loss_route_angle, label="train_final_loss_route_angle")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,3)
    plt.plot(range(len(train_final_loss_tl_dist)), train_final_loss_tl_dist, label="train_final_loss_tl_dist")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,4)
    plt.plot(range(len(train_final_loss_tl_state)), train_final_loss_tl_state, label="train_final_loss_tl_state")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.savefig("Affordance Prediction Training Loss Curves")

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(range(len(test_final_loss_lane_dist)), test_final_loss_lane_dist, label="test_final_loss_lane_dist")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,2)
    plt.plot(range(len(test_final_loss_route_angle)), test_final_loss_route_angle, label="test_final_loss_route_angle")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,3)
    plt.plot(range(len(test_final_loss_tl_dist)), test_final_loss_tl_dist, label="test_final_loss_tl_dist")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,4)
    plt.plot(range(len(test_final_loss_tl_state)), test_final_loss_tl_state, label="test_final_loss_tl_state")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.savefig("Affordance Prediction Validation Loss Curves")
    plt.show()


    plt.show()



def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/home/cangozpi/Desktop/train/mid" # TO BE CHANGED !
    val_root = "/home/cangozpi/Desktop/train/tt" # TO BE CHANGED !
    #train_root = "/userfiles/eozsuer16/expert_data/train" # TO BE CHANGED !
    #val_root = "/userfiles/eozsuer16/expert_data/val" # TO BE CHANGED !
    
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
