import gradio as gr
from fastai.learner import load_learner
from PIL import Image

learn = load_learner("export.pkl")


def classify_image(img):
    pred, pred_idx, probs = learn.predict(img)

    return {
        str(learn.dls.vocab[i]): float(probs[i])
        for i in range(len(probs))
    }


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Pet Breed Classifier",
    description="Upload an image of a pet."
)

demo.launch()