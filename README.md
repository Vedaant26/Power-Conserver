# 🏠 Smart Home Builder + EnerGenie ⚡  
**AI-Powered Smart Home Design & Energy Optimization**

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 🌍 Overview

**Smart Home Builder + EnerGenie** is a powerful web-based platform that transforms static house blueprints into intelligent, sustainable, and energy-efficient smart homes using **AI**, **machine learning**, **OCR**, and **real-time sensor data**.

From identifying rooms in architectural plans to recommending sensors and optimizing HVAC systems, EnerGenie enables users to **save energy**, **reduce carbon footprint**, and **design smarter homes** with zero technical knowledge.

---

## 🚀 Key Features

- 📄 **Blueprint Upload**: Upload floor plans (PDF or image) to begin.
- 🔍 **Room Detection via OCR**: Automatically identify and classify room labels (e.g., Kitchen, Bedroom).
- 🧠 **ML-Based Energy Prediction**: Estimate energy consumption based on room type, size, and usage patterns.
- 🔌 **Smart Sensor Recommendation**: Personalized sensor suggestions for each room (e.g., gas, temp, motion, humidity).
- 🌡️ **HVAC Load Estimation**: ML models predict and optimize heating/cooling needs per zone.
- ♻️ **Sustainability Insights**: View potential energy savings and carbon footprint reduction.
- 📊 **Visualized Output**: Interactive floor plan with suggested sensors and estimated energy usage.

---

## 💡 Use Cases

- 🏠 **Homeowners & Renters** – Reduce energy bills and automate homes affordably.
- 🏗️ **Architects & Builders** – Integrate energy optimization early in the design phase.
- 🌿 **Green Advocates** – Track and reduce environmental impact at a room-by-room level.
- 📡 **IoT Enthusiasts** – Build and manage smarter spaces with sensor-driven intelligence.

---

## 🧠 Tech Stack

| Technology       | Description                                    |
|------------------|------------------------------------------------|
| **Frontend**     | React.js, Tailwind CSS                        |
| **Backend**      | Node.js, Express.js, Python (for ML modules) |
| **ML Models**    | Scikit-learn, TensorFlow, custom ResNet       |
| **OCR**          | Tesseract.js / EasyOCR                        |
| **Sensors**      | ESP32 / IoT APIs (temp, motion, humidity)     |
| **Database**     | Firebase / MongoDB                            |
| **Deployment**   | Vercel / Render / Heroku                      |

---

## 🔍 How It Works

1. **Upload Floor Plan**
   - User uploads blueprint (image or PDF).
   - OCR + image processing extracts room labels and layout.

2. **Room Classification**
   - ML model classifies each room by label, size, and layout.

3. **Sensor Mapping**
   - Based on room type and usage, the system recommends relevant smart sensors.

4. **Energy Prediction**
   - ML predicts energy usage and suggests HVAC zones and automation rules.

5. **Output**
   - An interactive visualization shows energy data, savings, and environmental impact.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/smart-home-energenie.git
cd smart-home-energenie
npm install          # for frontend
cd backend
pip install -r requirements.txt  # for backend ML modules
# Frontend
npm run dev

cd backend
python app.py
