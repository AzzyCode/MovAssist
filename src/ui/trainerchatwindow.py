from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QLabel
import threading


class TrainerChatWindow(QWidget):
    def __init__(self, trainer, app_data=None):
        super().__init__()
        self.setWindowTitle("Chat with Your AI Trainer")
        self.resize(400, 500)
        self.trainer = trainer
        self.app_data = app_data
        
        layout = QVBoxLayout()
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        layout.addWidget(QLabel("Ask Your AI Trainer: "))
        layout.addWidget(self.chat_display)
        layout.addWidget(self.chat_input)
        layout.addWidget(self.send_button)
        self.setLayout(layout)
        
    
    def send_message(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        self.chat_display.append(f"Your: {question}")
        self.chat_input.clear()
        threading.Thread(target=self.get_trainer_response, args=(question,), daemon=True).start()
        
    
    def get_trainer_response(self, question):
        response = self.trainer.get_response(question, self.app_data)
        self.chat_display.append(f"Trainer: {response}")