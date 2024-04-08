import torch
import torchvision.transforms.v2 as transforms
import gradio as gr

from models.swin_transformer import SwinTransformer


def image_classifier(image):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ConvertImageDtype(torch.float),
        ]
    )

    image = transform(image)
    checkpoint = torch.load("runs/version_1/checkpoints/best.ckpt")
    model = SwinTransformer(num_classes=102)
    model.load_state_dict(checkpoint["state_dict"])

    outputs = model(image.unsqueeze(0))
    probs = torch.nn.functional.softmax(outputs, dim=-1)
    top_probs, top_ixs = probs[0].topk(10)

    predictions = {}
    for i, (ix_, prob_) in enumerate(zip(top_ixs, top_probs)):
        ix = ix_.item()
        prob = prob_.item()
        cls = label_names[ix].strip()
        # print(f"{i}: {cls:<45} --- {prob:.4f}")
        predictions.update({cls: prob})

    return predictions


if __name__ == "__main__":
    gr.Interface(
        title="Pytorch Trainer",
        fn=image_classifier,
        inputs="image",
        outputs="label",
        allow_flagging="never",
    ).launch()


label_names = [
    "alpine sea holly",
    "buttercup",
    "fire lily",
    "anthurium",
    "californian poppy",
    "foxglove",
    "artichoke",
    "camellia",
    "frangipani",
    "azalea",
    "canna lily",
    "fritillary",
    "ball moss",
    "canterbury bells",
    "garden phlox",
    "balloon flower",
    "cape flower",
    "gaura",
    "barbeton daisy",
    "carnation",
    "gazania",
    "bearded iris",
    "cautleya spicata",
    "geranium",
    "bee balm",
    "clematis",
    "giant white arum lily",
    "bird of paradise",
    "colt's foot",
    "globe thistle",
    "bishop of llandaff",
    "columbine",
    "globe-flower",
    "black-eyed susan",
    "common dandelion",
    "grape hyacinth",
    "blackberry lily",
    "corn poppy",
    "great masterwort",
    "blanket flower",
    "cyclamen ",
    "hard-leaved pocket orchid",
    "bolero deep blue",
    "daffodil",
    "hibiscus",
    "bougainvillea",
    "desert-rose",
    "hippeastrum ",
    "bromelia",
    "english marigold",
    "japanese anemone",
    "king protea",
    "peruvian lily",
    "stemless gentian",
    "lenten rose",
    "petunia",
    "sunflower",
    "lotus",
    "pincushion flower",
    "sweet pea",
    "love in the mist",
    "pink primrose",
    "sweet william",
    "magnolia",
    "pink-yellow dahlia?",
    "sword lily",
    "mallow",
    "poinsettia",
    "thorn apple",
    "marigold",
    "primula",
    "tiger lily",
    "mexican aster",
    "prince of wales feathers",
    "toad lily",
    "mexican petunia",
    "purple coneflower",
    "tree mallow",
    "monkshood",
    "red ginger",
    "tree poppy",
    "moon orchid",
    "rose",
    "trumpet creeper",
    "morning glory",
    "ruby-lipped cattleya",
    "wallflower",
    "orange dahlia",
    "siam tulip",
    "water lily",
    "osteospermum",
    "silverbush",
    "watercress",
    "oxeye daisy",
    "snapdragon",
    "wild pansy",
    "passion flower",
    "spear thistle",
    "windflower",
    "pelargonium",
    "spring crocus",
    "yellow iris",
]
