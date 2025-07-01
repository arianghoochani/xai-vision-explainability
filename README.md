# xai-vision-explainability
Explainable AI in Vision: Grad-CAM, Guided Grad-CAM, and TCAV implementations with slides from a University of Vienna talk.

## 🔍 Methods and Corresponding Resources

### 1. Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights important regions in an image that contribute to a model's prediction.

- 📓 Notebook: [`notebooks/grad_cam.ipynb`](notebooks/grad_cam.ipynb)
- 🖼️ Input Images: [`data/grad_cam_images/`](data/grad_cam_images/)
- Examples include cats, dogs, and other animals to visualize how the model localizes discriminative features.

---

### 2. Guided Grad-CAM

Guided Grad-CAM provides a finer-grained, more detailed version of Grad-CAM using guided backpropagation.

- 📓 Notebook: [`notebooks/guided_gradcam.ipynb`](notebooks/guided_gradcam.ipynb)
- 🖼️ Input Images: [`data/guided_gradcam_images/`](data/guided_gradcam_images/)
- Uses similar animal images to show enhanced pixel-level saliency.

---

### 3. TCAV (Testing with Concept Activation Vectors)

TCAV explains model predictions using **human-understandable concepts**, rather than individual pixels.

- 📓 Notebook: [`notebooks/tcav.ipynb`](notebooks/tcav.ipynb)
- 🖼️ Concept Images:
  - Striped: [`data/tcav_concepts/striped/`](data/tcav_concepts/striped/)
  - Zigzag: [`data/tcav_concepts/zigzag/`](data/tcav_concepts/zigzag/)
  - Dotted: [`data/tcav_concepts/dotted/`](data/tcav_concepts/dotted/)
- 🎲 Random Control Sets:
  - [`data/tcav_concepts/random_sets/random_1/`](data/tcav_concepts/random_sets/random_1/)
  - [`random_2/`, `random_3/`] are also included as control comparisons.

This setup hdelps quantify how sensitive the model is to concepts like "striped" versus unrelated image clusters.

