{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'flask'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[5], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mflask\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m Flask, request, render_template, redirect, url_for\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mcv2\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'flask'"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, render_template, redirect, url_for\n",
    "import numpy as np\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "from ultralytics import YOLO\n",
    "import os\n",
    "\n",
    "# Initialize the Flask application\n",
    "app = Flask(__name__)\n",
    "\n",
    "# Load your YOLO model\n",
    "model = YOLO('runs/segment/train/weights/best.pt')\n",
    "\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/upload', methods=['POST'])\n",
    "def upload_image():\n",
    "    if 'file' not in request.files:\n",
    "        return redirect(request.url)\n",
    "    \n",
    "    file = request.files['file']\n",
    "    \n",
    "    if file.filename == '':\n",
    "        return redirect(request.url)\n",
    "\n",
    "    # Save the uploaded file temporarily\n",
    "    img_path = os.path.join('uploads', file.filename)\n",
    "    file.save(img_path)\n",
    "\n",
    "    # Process the image\n",
    "    result_image = process_image(img_path)\n",
    "\n",
    "    # Save the result image\n",
    "    result_path = os.path.join('static', 'result.png')\n",
    "    cv2.imwrite(result_path, result_image)\n",
    "\n",
    "    return render_template('result.html', result_image=result_path)\n",
    "\n",
    "def process_image(image_path):\n",
    "    img = cv2.imread(image_path)\n",
    "    results = model(img, imgsz=640, iou=0.8, conf=0.19)\n",
    "\n",
    "    classes = results[0].boxes.cls.cpu().numpy()\n",
    "    class_names = results[0].names\n",
    "    masks = results[0].masks.data\n",
    "    num_masks = masks.shape[0]\n",
    "\n",
    "    colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_masks)]\n",
    "    labeled_image = img.copy()\n",
    "\n",
    "    for i in range(num_masks):\n",
    "        color = colors[i]\n",
    "        mask = masks[i].cpu()\n",
    "        mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)\n",
    "        \n",
    "        class_index = int(classes[i])\n",
    "        class_name = class_names[class_index]\n",
    "        \n",
    "        mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "        cv2.drawContours(labeled_image, mask_contours, -1, color, 5)\n",
    "\n",
    "        if len(mask_contours) > 0:\n",
    "            text_position = (int(mask_contours[0][:, 0, 0].mean()), int(mask_contours[0][:, 0, 1].mean()))\n",
    "            cv2.putText(labeled_image, class_name, text_position, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)\n",
    "\n",
    "    return labeled_image\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
