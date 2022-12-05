import smtplib
import email.message
import email.utils

class Mailer():
    def __init__(self, send_to, subject):
        self.server = smtplib.SMTP("smtp.metu.edu.tr", 587)
        send_from = "xxx@metu.edu.tr"
        self.msg = self.create_message(send_from, send_to, subject)

    def __call__(self, message=''):
        self.msg.set_payload(message)
        self.send(self.msg)

    def send(self, msg):
        self.server.starttls()
        self.server.login('xxx', 'xxx')
        self.server.sendmail(msg['From'], [msg['To']], msg.as_string())
        self.server.quit()

    def create_message(self, send_from, send_to, subject):
        msg = email.message.Message()
        msg['From'] = send_from
        msg['To'] = send_to
        msg['Subject'] = subject
        msg.add_header('Content-Type', 'text')
        return msg

# mailer = Mailer("abyesilyurt@gmail.com", 'test')
# mailer('hi')

