import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset
from models.cilrs import CILRS
#imports below are added by me
import matplotlib.pyplot as plt


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    #print("new test set evaluation started !")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    loss_f = torch.nn.L1Loss()

    # validate in minibatches
    with torch.no_grad():
        final_cum_loss = 0
        final_speed_loss = 0
        final_throttle_loss = 0
        final_brake_loss = 0
        final_steer_loss = 0
        for img, measurements in dataloader:
            # Extract input features/targets from measurements
            command, speed, expert_throttle, expert_brake, expert_steer, \
            _, _, _, _, _, _, _, _, _ = measurements
            
            if device != "cpu":
                command = command.cuda()
                speed = speed.cuda()
                expert_throttle = expert_throttle.cuda()
                expert_brake = expert_brake.cuda()
                expert_steer = expert_steer.cuda()
                img = img.cuda()
            

            # Forward pass 
            pred_speed, pred_throttle, pred_brake, pred_steer = model(img.float(), command, speed.unsqueeze(1))
        
            # Calculate losses:
            # calculate speed loss
            cur_speed_loss = loss_f(pred_speed, speed)
            cur_throttle_loss = loss_f(pred_throttle, expert_throttle)
            cur_brake_loss = loss_f(pred_brake, expert_brake)
            cur_steer_loss = loss_f(pred_steer, expert_steer)
            
            cur_cum_loss = cur_speed_loss + cur_throttle_loss + cur_brake_loss + cur_steer_loss 
            
            # record training loss / convert mean batch losses to total losses
            batch_size = len(img)
            final_speed_loss += (cur_speed_loss.item() * batch_size)
            final_throttle_loss += (cur_throttle_loss.item() * batch_size)
            final_brake_loss += (cur_brake_loss.item() * batch_size)
            final_steer_loss += (cur_steer_loss.item() * batch_size)
            final_cum_loss += (cur_cum_loss.item() * batch_size)
            
        # Calculate average loss for the current epoch
        final_speed_loss /= len(dataloader.dataset)
        final_throttle_loss /= len(dataloader.dataset)
        final_brake_loss /= len(dataloader.dataset)
        final_steer_loss /= len(dataloader.dataset)
        final_cum_loss /= len(dataloader.dataset)

        #print("Average validation loss for the last epoch: ", final_cum_loss)
        return final_speed_loss, final_throttle_loss, final_brake_loss, final_steer_loss, final_cum_loss
    

def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()
    # Training config
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    loss_f = torch.nn.L1Loss()

    # Train in minibatches
    final_cum_loss = 0
    final_speed_loss = 0
    final_throttle_loss = 0
    final_brake_loss = 0
    final_steer_loss = 0
    for i,(img, measurements) in enumerate(dataloader):
        # if i % 500 == 0:
        #     print("Batch: ", i, "/", len(dataloader))
        # Extract input features/targets from measurements
        command, speed, expert_throttle, expert_brake, expert_steer, \
            _, _, _, _, _, _, _, _, _ = measurements
        
        if device != "cpu":
            command = command.cuda()
            speed = speed.cuda()
            expert_throttle = expert_throttle.cuda()
            expert_brake = expert_brake.cuda()
            expert_steer = expert_steer.cuda()
            img = img.cuda() # [Batch_size, C, H , W]
        
        # Forward pass 
        pred_speed, pred_throttle, pred_brake, pred_steer = model(img.float(), command, speed.unsqueeze(1)) # [Batch_size]
        
        # Calculate losses:
        # calculate speed loss
        cur_speed_loss = loss_f(pred_speed, speed)
        cur_throttle_loss = loss_f(pred_throttle, expert_throttle)
        cur_brake_loss = loss_f(pred_brake, expert_brake)
        cur_steer_loss = loss_f(pred_steer, expert_steer)
        
        cur_cum_loss = cur_speed_loss + cur_throttle_loss + cur_brake_loss + cur_steer_loss 
         
        
        
        # Backpropagate
        optimizer.zero_grad()
        cur_cum_loss.backward()
        optimizer.step()

        # record training loss / convert mean batch losses to total losses
        batch_size = len(img)
        final_speed_loss += (cur_speed_loss.item() * batch_size)
        final_throttle_loss += (cur_throttle_loss.item() * batch_size)
        final_brake_loss += (cur_brake_loss.item() * batch_size)
        final_steer_loss += (cur_steer_loss.item() * batch_size)
        final_cum_loss += (cur_cum_loss.item() * batch_size)
        
    # Calculate average loss for the current epoch
    final_speed_loss /= len(dataloader.dataset)
    final_throttle_loss /= len(dataloader.dataset)
    final_brake_loss /= len(dataloader.dataset)
    final_steer_loss /= len(dataloader.dataset)
    final_cum_loss /= len(dataloader.dataset)

    print("Average training loss for the last epoch: ", final_cum_loss)
    return final_speed_loss, final_throttle_loss, final_brake_loss, final_steer_loss, final_cum_loss






def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    train_speed_loss, train_throttle_loss, train_brake_loss, train_steer_loss, train_cum_loss = [], [], [], [], []
    val_speed_loss, val_throttle_loss, val_brake_loss, val_steer_loss, val_cum_loss = [], [], [], [], []
    
    for e in train_loss:
        train_speed_loss.append(e[0])
        train_throttle_loss.append(e[1])
        train_brake_loss.append(e[2])
        train_steer_loss.append(e[3])
        train_cum_loss.append(e[4])

    for e in val_loss:
        val_speed_loss.append(e[0])
        val_throttle_loss.append(e[1])
        val_brake_loss.append(e[2])
        val_steer_loss.append(e[3])
        val_cum_loss.append(e[4])


    plt.figure()
    
    plt.subplot(1,2,1)
    plt.plot(range(len(train_speed_loss)), train_speed_loss, label="train_speed_loss")
    plt.plot(range(len(train_throttle_loss)), train_throttle_loss, label="train_throttle_loss")
    plt.plot(range(len(train_brake_loss)), train_brake_loss, label="train_brake_loss")
    plt.plot(range(len(train_steer_loss)), train_steer_loss, label="train_steer_loss")
    plt.plot(range(len(train_cum_loss)), train_cum_loss, label="train_cum_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.title("Training Loss Curve")
    

    plt.subplot(1,2,2)
    plt.plot(range(len(val_speed_loss)), val_speed_loss, label="val_speed_loss")
    plt.plot(range(len(val_throttle_loss)), val_throttle_loss, label="val_throttle_loss")
    plt.plot(range(len(val_brake_loss)), val_brake_loss, label="val_brake_loss")
    plt.plot(range(len(val_steer_loss)), val_steer_loss, label="val_steer_loss")
    plt.plot(range(len(val_cum_loss)), val_cum_loss, label="val_cum_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.title("Validation Loss Curve")
    plt.savefig("CILRS Training and Validation Loss Curves")

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(range(len(train_speed_loss)), train_speed_loss, label="train_speed_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    
    plt.subplot(2,2,2)
    plt.plot(range(len(train_throttle_loss)), train_throttle_loss, label="train_throttle_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,3)
    plt.plot(range(len(train_brake_loss)), train_brake_loss, label="train_brake_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,4)
    plt.plot(range(len(train_steer_loss)), train_steer_loss, label="train_steer_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")
    plt.savefig("CILRS Training Loss Curves")

    plt.figure()

    plt.subplot(2,2,1)
    plt.plot(range(len(val_speed_loss)), val_speed_loss, label="val_speed_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,2)
    plt.plot(range(len(val_throttle_loss)), val_throttle_loss, label="val_throttle_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,3)
    plt.plot(range(len(val_brake_loss)), val_brake_loss, label="val_brake_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.subplot(2,2,4)
    plt.plot(range(len(val_steer_loss)), val_steer_loss, label="val_steer_loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("average loss")

    plt.savefig("CILRS Validation Loss Curves")
    plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/home/cangozpi/Desktop/train/all/train" # TO BE CHANGED !
    val_root = "/home/cangozpi/Desktop/train/all/val" # TO BE CHANGED !
    #train_root = "/userfiles/eozsuer16/expert_data/train" # TO BE CHANGED !
    #val_root = "/userfiles/eozsuer16/expert_data/val" # TO BE CHANGED !
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"

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
