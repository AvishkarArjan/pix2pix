import torch
import config
from torchvision.utils import save_image
from PIL import Image


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#

        # y_fake = Image.fromarray(y_fake[0])
        # y_fake.save(folder+f"/y_gen_{epoch}.png")

        # x = Image.fromarray((x * 0.5 + 0.5)[0])
        # x.save(folder + f"/input_{epoch}.png")
        save_image(y_fake[0], folder + f"/y_gen_{epoch}.png")
        x = x * 0.5 + 0.5
        save_image(x[0], folder + f"/input_{epoch}.png")
        if epoch == 1:
            # y = Image.fromarray((y * 0.5 + 0.5)[0])
            # y.save(folder + f"/label_{epoch}.png")
            y = y * 0.5 + 0.5
            save_image(y[0], folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

