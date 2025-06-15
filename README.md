# Rock Paper Scissors - AI Game 🎮✊✋✌️

Play Rock Paper Scissors against an AI using your webcam!  
Built with Python, OpenCV, and a custom-trained neural network.

> ⚙️ Originally inspired by [SouravJohar](https://github.com/SouravJohar)'s repo — customized and experimented by me while learning Computer Vision and ML.

## 🚀 Features
- Webcam-based gesture detection
- AI model to recognize Rock / Paper / Scissors / Nothing
- Real-time game interaction

## 🧠 Tech Stack
- Python 3
- TensorFlow + Keras
- OpenCV

## 🔧 How to Use

1. **Clone this repo**  
```bash
git clone https://github.com/your-username/rock-paper-scissors-.git
cd rock-paper-scissors


2. Install the dependencies
```sh
$ pip install -r requirements.txt
```

3. Gather Images for each gesture (rock, paper and scissors and None):
In this example, we gather 200 images for the "rock" gesture
```sh
$ python3 gather_images.py rock 200
```

4. Train the model
```sh
$ python3 train.py
```

5. Test the model on some images
```sh
$ python3 test.py <path_to_test_image>
```

6. Play the game with your computer!
```sh
$ python3 play.py
```
