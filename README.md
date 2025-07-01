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

TCAV explains model predictions based on human-interpretable **concepts**, enabling us to answer:  
*"How sensitive is this model to the concept of stripes when identifying zebras?"*

- 📓 Notebook: [`notebooks/tcav.ipynb`](notebooks/tcav.ipynb)

#### 🧠 Concept Images:
Located in [`data/tcav_concepts/concepts/`](data/tcav_concepts/concepts/):
- 🟦 **Striped** – [`striped/`](data/tcav_concepts/concepts/striped/)
- 🟧 **Zigzag** – [`zigzag/`](data/tcav_concepts/concepts/zigzag/)
- 🟨 **Dotted** – [`dotted/`](data/tcav_concepts/concepts/dotted/)

#### 🎲 Random Control Sets:
Located in [`data/tcav_concepts/random_sets/`](data/tcav_concepts/random_sets/):
- [`random_1/`](data/tcav_concepts/random_sets/random_1/)
- [`random_2/`](data/tcav_concepts/random_sets/random_2/)
- [`random_3/`](data/tcav_concepts/random_sets/random_3/)

These are used as neutral controls to contrast with the meaningful concepts.

#### 🦓 Target Class Images:
Located in [`data/tcav_concepts/target_class/zebra/`](data/tcav_concepts/target_class/zebra/):
- Contains sample **zebra images** used as the target class for TCAV sensitivity analysis.

---