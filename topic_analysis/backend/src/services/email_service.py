"""
Email service for sending transactional emails.
"""
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails."""
    
    def __init__(self):
        self.api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@trendanalysis.com")
        self.from_name = os.getenv("FROM_NAME", "Trend Analysis Platform")
        self.frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.enabled = bool(self.api_key)
        
        # Initialize SendGrid client if API key is available
        if self.enabled:
            try:
                from sendgrid import SendGridAPIClient
                self.client = SendGridAPIClient(api_key=self.api_key)
                logger.info("Email service initialized with SendGrid")
            except ImportError:
                logger.warning("SendGrid not installed, email service will log only")
                self.client = None
                self.enabled = False
        else:
            logger.warning("No SendGrid API key provided, email service will log only")
            self.client = None
    
    def send_email(self, to_email: str, subject: str, html_content: str, text_content: str = None) -> bool:
        """Send an email."""
        if not self.enabled:
            logger.info(f"Email service disabled. Would send to {to_email}: {subject}")
            return True
        
        try:
            from sendgrid.helpers.mail import Mail
            
            message = Mail(
                from_email=(self.from_email, self.from_name),
                to_emails=to_email,
                subject=subject,
                html_content=html_content,
                plain_text_content=text_content
            )
            
            response = self.client.send(message)
            success = response.status_code in [200, 201, 202]
            
            if success:
                logger.info(f"Email sent successfully to {to_email}")
            else:
                logger.error(f"Failed to send email to {to_email}, status: {response.status_code}")
            
            return success
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def send_verification_email(self, to_email: str, verification_token: str, user_name: str) -> bool:
        """Send email verification email."""
        verification_url = f"{self.frontend_url}/verify-email?token={verification_token}"
        
        subject = "Verify Your Email - Trend Analysis Platform"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Email Verification</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .button {{ display: inline-block; background-color: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Trend Analysis Platform!</h1>
                </div>
                <div class="content">
                    <p>Hi {user_name},</p>
                    <p>Thank you for registering with Trend Analysis Platform. To complete your registration, please verify your email address by clicking the button below:</p>
                    <p style="text-align: center;">
                        <a href="{verification_url}" class="button">Verify Email Address</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background-color: #eee; padding: 10px; border-radius: 3px;">
                        {verification_url}
                    </p>
                    <p><strong>This link will expire in 24 hours.</strong></p>
                    <p>If you didn't create an account with us, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>© 2024 Trend Analysis Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to Trend Analysis Platform!
        
        Hi {user_name},
        
        Thank you for registering with Trend Analysis Platform. To complete your registration, please verify your email address by visiting the following link:
        
        {verification_url}
        
        This link will expire in 24 hours.
        
        If you didn't create an account with us, please ignore this email.
        
        Best regards,
        Trend Analysis Platform Team
        """
        
        return self.send_email(to_email, subject, html_content, text_content)
    
    def send_password_reset_email(self, to_email: str, reset_token: str, user_name: str) -> bool:
        """Send password reset email."""
        reset_url = f"{self.frontend_url}/reset-password?token={reset_token}"
        
        subject = "Reset Your Password - Trend Analysis Platform"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Password Reset</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f44336; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .button {{ display: inline-block; background-color: #f44336; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 3px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Password Reset Request</h1>
                </div>
                <div class="content">
                    <p>Hi {user_name},</p>
                    <p>You requested to reset your password for your Trend Analysis Platform account. Click the button below to reset your password:</p>
                    <p style="text-align: center;">
                        <a href="{reset_url}" class="button">Reset Password</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; background-color: #eee; padding: 10px; border-radius: 3px;">
                        {reset_url}
                    </p>
                    <div class="warning">
                        <p><strong>Important:</strong></p>
                        <ul>
                            <li>This link will expire in 1 hour</li>
                            <li>If you didn't request this password reset, please ignore this email</li>
                            <li>Your password will not be changed until you click the link above</li>
                        </ul>
                    </div>
                </div>
                <div class="footer">
                    <p>© 2024 Trend Analysis Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Password Reset Request
        
        Hi {user_name},
        
        You requested to reset your password for your Trend Analysis Platform account. Visit the following link to reset your password:
        
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you didn't request this password reset, please ignore this email.
        
        Best regards,
        Trend Analysis Platform Team
        """
        
        return self.send_email(to_email, subject, html_content, text_content)
    
    def send_welcome_email(self, to_email: str, user_name: str) -> bool:
        """Send welcome email after successful verification."""
        subject = "Welcome to Trend Analysis Platform!"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2196F3; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; background-color: #f9f9f9; }}
                .button {{ display: inline-block; background-color: #2196F3; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Trend Analysis Platform!</h1>
                </div>
                <div class="content">
                    <p>Hi {user_name},</p>
                    <p>Congratulations! Your email has been verified and your account is now active.</p>
                    <p>You can now start using all the features of our platform:</p>
                    <ul>
                        <li>Research trending topics and keywords</li>
                        <li>Analyze market trends and opportunities</li>
                        <li>Generate content ideas and strategies</li>
                        <li>Track your content performance</li>
                    </ul>
                    <p style="text-align: center;">
                        <a href="{self.frontend_url}/dashboard" class="button">Go to Dashboard</a>
                    </p>
                    <p>If you have any questions, feel free to contact our support team.</p>
                </div>
                <div class="footer">
                    <p>© 2024 Trend Analysis Platform. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        Welcome to Trend Analysis Platform!
        
        Hi {user_name},
        
        Congratulations! Your email has been verified and your account is now active.
        
        You can now start using all the features of our platform:
        - Research trending topics and keywords
        - Analyze market trends and opportunities
        - Generate content ideas and strategies
        - Track your content performance
        
        Visit your dashboard: {self.frontend_url}/dashboard
        
        If you have any questions, feel free to contact our support team.
        
        Best regards,
        Trend Analysis Platform Team
        """
        
        return self.send_email(to_email, subject, html_content, text_content)
    
    def generate_verification_token(self) -> str:
        """Generate a verification token."""
        return str(uuid.uuid4())
    
    def generate_reset_token(self) -> str:
        """Generate a password reset token."""
        return str(uuid.uuid4())
    
    def is_enabled(self) -> bool:
        """Check if email service is enabled."""
        return self.enabled
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get email service status information."""
        return {
            "enabled": self.enabled,
            "provider": "SendGrid" if self.enabled else "Logging Only",
            "from_email": self.from_email,
            "from_name": self.from_name,
            "frontend_url": self.frontend_url
        }

# Create singleton instance
email_service = EmailService()
