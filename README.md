# Sportsman Classifier

This is a web-based application that classifies uploaded images into well-known sports personalities. The application is built using **Next.js** with **React** for the frontend and integrates with a **Flask API** for image classification.

## Features
- 🏆 **Upload an image** to classify a sports personality.
- 📊 **Displays a structured table** of classification results with confidence scores.
- 🎯 **Highlights the predicted class** with the highest probability.
- 🔄 **Drag & Drop support** for easy image uploads.
- 🚀 **Fast processing** with a Flask-based backend.

## Technologies Used
### **Frontend:**
- **Next.js** (React Framework)
- **Tailwind CSS** (for styling)
- **react-dropzone** (for image uploads)
- **Sonner** (for toast notifications)

### **Backend:**
- **Flask** (Python API for image classification)
- **OpenCV / ML Model** (for classification logic)
- **Pywavlet / ML Model** (for classification logic)
- **Numpy / ML Model** (for classification logic)

## Athlete Classification Order
The classification is based on predefined sports personalities:

| Athlete | Class Index |
|---------|------------|
| Lionel Messi | 0|
| Maria Sharapova | 1|
| Roger Federer | 2 |
| Serena Williams | 3 |
| Virat Kohli | 4|

## How It Works
1. 🖼️ Upload an image using **Drag & Drop** or **File Selector**.
2. 🔄 Click the **"Classify"** button.
3. 🏆 The backend processes the image and returns a **confidence score** for each athlete.
4. 📊 Results are displayed in a **table format**, with the **predicted class highlighted**.


## API Endpoint
- **POST /classify_image** → Accepts an image and returns classification results in JSON format.

## Example Response
```json
{
  "class": "virat_kohli",
  "class_prob": [
    1.47,  // Cropped
    1.17,  // Lionel Messi
    1.02,  // Maria Sharapova
    1.08,  // Roger Federer
    95.24  // Serena Williams
  ]
}
```


