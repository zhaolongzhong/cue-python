import os
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()


def send_email(content, email_list: list[str]):
    """
    Send email to email_list, sender is the email account in environment
    Only need to set EMAIL_USERNAME and EMAIL_APP_PASSWORD in environment
    """
    # creates SMTP session
    s = smtplib.SMTP("smtp.gmail.com", 587)
    # s = smtplib.SMTP('smtp.gmail.com', 465)

    # start TLS for security
    s.starttls()

    # Authentication
    username = os.environ.get("EMAIL_USERNAME")
    # Get app password: Account -> Security -> 2-Step Verification -> App Password
    # https://myaccount.google.com/apppasswords
    password = os.environ.get("EMAIL_APP_PASSWORD")
    if username is None or password is None:
        raise ValueError("EMAIL_USERNAME or EMAIL_APP_PASSWORD not set in environment")

    s.login(username, password)

    sender = username
    recivers = email_list

    # subject and message to be sent
    subject = "Newsletter"
    msg = MIMEText(content, "html")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recivers)

    # sending the mail
    s.sendmail(sender, recivers, msg.as_string())

    # terminating the session
    s.quit()


def main():
    # uv run python src/utils/email_utils.py
    send_email("Hello World", ["hello@example.com"])


if __name__ == "__main__":
    main()
