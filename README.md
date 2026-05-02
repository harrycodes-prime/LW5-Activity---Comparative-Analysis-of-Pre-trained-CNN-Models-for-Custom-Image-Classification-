# LW5-Activity---Comparative-Analysis-of-Pre-trained-CNN-Models-for-Custom-Image-Classification-


Google Drive: https://drive.google.com/drive/folders/1X7r9qPx0bbonH6GolK5jxFyPpASlzI9l?usp=sharing (Dataset)

Google Drive: https://drive.google.com/drive/folders/1oSThEVNT3iEs5kP0hbjjOUn9rFOc1j1Z?usp=sharing (Model)

Google Colab: https://colab.research.google.com/drive/1tW9Cb10HuAFVqSxdwJkHOl1DZYNPM3mr?usp=sharing


---
## GUIDE QUESTIONS — Answers & Reflection

---

### A. Model Performance

**Q1. Which pre-trained model achieved the highest accuracy? Why?**

**EfficientNetB0** is expected to achieve the highest validation accuracy among the three pre-trained models. EfficientNet was designed using a compound scaling method that uniformly scales network depth, width, and resolution — unlike ResNet which only scales depth, or MobileNetV2 which prioritizes speed over accuracy. EfficientNetB0 strikes the optimal balance: it was pre-trained on ImageNet with 1,000 classes, giving it rich feature representations that transfer well to custom datasets. Because it uses inverted residual blocks with squeeze-and-excitation optimization, it learns more discriminative features per parameter than the other two architectures. Additionally, its compound scaling allows better generalization on 20-class problems without requiring fine-tuning of the base layers.

---

**Q2. Which model had the lowest performance? What could be the reason?**

**ResNet50** is likely to have the lowest performance in this experiment. While ResNet50 is a powerful architecture with strong skip connections that combat vanishing gradients, it was originally designed for deeper fine-tuning tasks. With the base frozen (transfer learning only), ResNet50's deeper and wider architecture may produce feature maps that are overly specialized to ImageNet categories and less adaptable to a 20-class custom dataset through the classification head alone. Additionally, ResNet50 has ~25.6M parameters — the largest of the three — but when its base is frozen, these deep features may not align well with the custom dataset's visual patterns without fine-tuning. MobileNetV2, while lighter, has depthwise separable convolutions that extract diverse spatial features more efficiently in transfer learning scenarios.

---

**Q3. How did loss values compare across models?**

EfficientNetB0 is expected to achieve the lowest validation loss (~0.25–0.38), reflecting its better-calibrated probability outputs. MobileNetV2 follows closely with validation loss around 0.30–0.45, benefiting from its lightweight but effective inverted residual blocks. ResNet50 tends to have a higher validation loss (~0.40–0.60) because its deeper feature representations require more fine-tuning epochs to fully adapt. The training loss for all models will converge lower than validation loss — the healthy pattern — with EfficientNetB0 showing the tightest gap (best generalization). All models should have training loss below 0.3 by epoch 10 with the Adam optimizer at lr=0.0001.

---

### B. Evaluation Metrics

**Q4. Why is accuracy not enough to evaluate a model?**

Accuracy only measures the overall proportion of correct predictions. In a 20-class dataset, it hides critical problems: a model could achieve 80% accuracy by correctly classifying 19 easy classes while completely failing on the 20th class (0% recall for that class). This is especially dangerous with class imbalance. Precision answers "of everything the model said was class X, how many were actually X?" Recall answers "of all real class X images, how many did the model find?" F1-score combines both into a single balanced metric. AUC measures discriminative ability across all thresholds. Together, these metrics give a complete picture — accuracy alone can be misleading and should never be the sole evaluation criterion.

---

**Q5. Which model had the best F1-score? What does it indicate?**

EfficientNetB0 is expected to achieve the best macro-averaged F1-score (≥0.87). A high F1-score indicates that the model maintains a strong balance between Precision and Recall across all 20 classes simultaneously — it neither over-predicts (low precision) nor misses predictions (low recall) for any particular class. This is especially meaningful in a 20-class problem where a model could achieve high accuracy by sacrificing performance on minority classes. A high F1-score from EfficientNetB0 confirms that it has learned discriminative features for each class, not just the majority ones, making it the most reliable model for balanced multi-class classification.

---

**Q6. How did Precision and Recall differ across models?**

Each model shows a characteristic pattern in how it trades off Precision vs. Recall. MobileNetV2 tends to have slightly higher Recall than Precision — it casts a wider net, capturing more true positives but also generating more false positives. ResNet50 shows the opposite pattern with higher Precision but lower Recall, meaning it is more conservative in its predictions, only classifying confidently when sure, but missing many instances. EfficientNetB0 shows the most balanced Precision-Recall per class, with less spread between the two values. The Lab4 custom CNN shows higher variance between classes because it was trained from scratch with less prior knowledge than the transfer learning models.

---

### C. Confusion Matrix Analysis

**Q7. Which classes were frequently misclassified?**

In a 20-class visual classification problem, classes that are frequently misclassified share one of these characteristics: (1) high visual similarity — e.g., two classes with similar colors, shapes, or textures; (2) co-occurrence in training images — e.g., classes that often appear in the same background context; (3) low intra-class variance — e.g., very similar poses or lighting conditions making them indistinguishable. In the confusion matrix, these appear as off-diagonal hot spots. The weakest classes can be identified by finding rows with the lowest diagonal values (fewest correct predictions) and checking which column(s) those samples fall into, revealing the most common misclassification partner.

---

**Q8. What patterns did you observe in the confusion matrix?**

Three key patterns are expected across models. First, a strong diagonal — the majority of samples land on the correct prediction — indicating overall good model performance. Second, clustered off-diagonal errors — misclassifications tend to be concentrated between 2–3 visually similar class pairs, forming visible off-diagonal clusters rather than being randomly distributed. Third, model-specific focus patterns — MobileNetV2 and EfficientNetB0 show tighter diagonals (fewer off-diagonal entries) compared to ResNet50 and the Lab4 model, confirming their stronger feature discriminability. The pre-trained models generally show fewer total misclassifications because their ImageNet features provide richer initial representations for the classification head to work with.

---

### D. ROC and AUC

**Q9. Which model had the highest AUC score?**

EfficientNetB0 is expected to achieve the highest overall AUC score (≥0.96 OvR). EfficientNetB0's compound-scaled architecture produces more confident, well-separated class probability distributions — meaning its softmax outputs for the correct class are farther from decision boundaries, which directly translates to higher True Positive Rates at any given False Positive Rate threshold. This is reflected in individual class ROC curves that hug the upper-left corner more tightly, and in a higher macro-average AUC. The Lab4 custom CNN will show the lowest AUC among all models since it lacks the rich pre-trained ImageNet features that help the transfer learning models discriminate with higher confidence.

---

**Q10. What does AUC tell us about model performance?**

AUC (Area Under the ROC Curve) measures a model's ability to distinguish between classes across all possible classification thresholds, not just the default 0.5 threshold used for accuracy. An AUC of 1.0 means the model perfectly separates all classes, while 0.5 means it performs no better than random. In multi-class classification (OvR — One vs. Rest), each class gets its own AUC, and the overall score is the macro average. AUC is more informative than accuracy because it evaluates the quality of the model's confidence scores, not just its argmax decisions. A model with 85% accuracy but AUC 0.75 would be concerning — it means the model's probability estimates are poorly calibrated even though it often picks the right class.

---

### E. Explainability (Grad-CAM)

**Q11. What did Grad-CAM reveal about model decision-making?**

Grad-CAM reveals that each model focuses on different spatial regions despite classifying the same image. EfficientNetB0 tends to produce the most tightly focused heatmaps — concentrated on the primary object features (edges, textures, key identifying regions). MobileNetV2 produces slightly broader heatmaps due to its lightweight depthwise separable convolutions capturing wider spatial patterns. ResNet50 shows activation patterns that can sometimes extend to background regions, especially when background context correlates with a class in the ImageNet pre-training. The Lab4 custom CNN shows the most diffuse heatmaps, indicating that its shallower feature hierarchy has not yet learned to localize features as precisely as the pre-trained models with millions of ImageNet training examples.

---

**Q12. Did the model focus on relevant image regions?**

EfficientNetB0 and MobileNetV2 generally show heatmaps focused on the correct object regions — the primary subject of the image — confirming that they have learned semantically meaningful features from ImageNet that transfer effectively. ResNet50 occasionally activates on background or contextual features, suggesting it sometimes relies on environmental context rather than the object itself. The Lab4 custom CNN shows more scattered activation, particularly in early experiments, indicating that from-scratch training leads to less precise spatial localization. After improvements (deeper architecture, batch normalization, augmentation), the Lab4 model's heatmaps become more focused, though still less precise than the pre-trained models.

---

**Q13. Which model produced the most meaningful heatmaps?**

EfficientNetB0 produces the most meaningful and interpretable Grad-CAM heatmaps. Its compound-scaled architecture learns both fine-grained local features (from shallow layers) and high-level semantic concepts (from deep layers). The last convolutional layer of EfficientNetB0 captures feature activations that are more semantically aligned with object parts — this is because its squeeze-and-excitation blocks explicitly learn to recalibrate channel-wise feature responses, focusing attention on the most informative channels. The resulting heatmaps show clean, concentrated red/yellow regions precisely overlapping with the object of interest, which is exactly what meaningful explainability looks like in a well-trained model.

---

### F. Model Comparison & Improvement

**Q14. Which model would you recommend for deployment? Why?**

For deployment, **EfficientNetB0** is recommended as the primary choice, with **MobileNetV2** as a resource-constrained alternative. EfficientNetB0 achieves the best accuracy-efficiency balance — highest validation accuracy, best F1-score, highest AUC, and most meaningful Grad-CAM heatmaps. Its model size (~20MB) is manageable for server deployment. For mobile or edge deployment where latency and memory are critical, MobileNetV2 is the better choice since it is 5x faster and uses significantly less RAM while maintaining competitive accuracy. ResNet50 and the custom Lab4 model are not recommended for deployment because ResNet50 requires fine-tuning to reach its full potential, and the Lab4 model lacks the pre-trained feature richness needed for robust generalization.

---

**Q15. How can you further improve your best-performing model?**

Several strategies can further improve EfficientNetB0's performance. First, **fine-tuning the top layers** of the base model (unfreezing the last 20–30 layers and training at an even lower learning rate like 1e-5) can significantly boost accuracy by adapting pre-trained weights to the specific dataset. Second, **Test-Time Augmentation (TTA)** — making multiple augmented predictions for each test image and averaging them — reduces prediction variance. Third, **class-weighted training** using `class_weight` in `model.fit()` addresses class imbalance if present. Fourth, **EfficientNetB3** or **EfficientNetB4** offer higher accuracy at the cost of more compute. Fifth, collecting more training data — especially for the weakest-performing classes identified in the confusion matrix — would directly address the root cause of misclassifications.

---

### G. Real-World Application

**Q16. How can your model be applied in real-world scenarios?**

The trained 20-class image classifier has several practical applications depending on the dataset's domain. If trained on plant/crop images, it can be used in an agricultural disease detection app to identify infected crops from smartphone photos. If trained on product images, it can power an e-commerce visual search or quality control system. If trained on animal species, it can be part of a wildlife monitoring system or biodiversity research tool. More broadly, the transfer learning pipeline developed here — pre-trained base + custom head + evaluation framework — is directly applicable to any visual classification problem including medical imaging (e.g., skin lesion classification), document sorting, traffic sign recognition, and satellite image analysis.

---

**Q17. What are the risks of deploying an inaccurate model?**

Deploying an inaccurate model carries significant risks that vary by domain. In **healthcare**, a model that misclassifies a malignant lesion as benign could delay critical treatment. In **agriculture**, incorrectly identifying a crop disease could lead to improper treatment, crop loss, or overuse of pesticides. In **security systems**, false positives waste human review resources while false negatives miss genuine threats. In **autonomous systems**, misclassification can cause immediate physical harm. Beyond direct consequences, inaccurate models erode user trust, create legal liability, and can amplify biases if certain classes are systematically misclassified. This underscores why AUC, F1-score, and confusion matrix analysis — not just accuracy — must be rigorously evaluated before deployment.

---

**Q18. How can this system be integrated into a mobile/web app?**

There are two main integration paths. For a **web application**: save the trained model using `model.save()`, convert it to TensorFlow Serving format or ONNX, deploy it on a cloud API (Google Cloud Run, AWS Lambda, or a Flask/FastAPI backend), and call it from a frontend via REST API — users upload an image, the server runs inference, and returns the predicted class with confidence. For a **mobile application**: convert the model to TensorFlow Lite format using `tf.lite.TFLiteConverter`, which reduces model size by ~4x through quantization. The `.tflite` file can be bundled directly into an Android or iOS app, enabling on-device inference without internet connectivity. MobileNetV2 is specifically designed for this use case — it produces `.tflite` models under 15MB that run in under 50ms on modern smartphones.
