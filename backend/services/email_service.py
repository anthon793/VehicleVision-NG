"""
Email Service for sending stolen vehicle alerts.
Sends email notifications to admins when a stolen vehicle is detected.
"""

import smtplib
import ssl
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, List
from datetime import datetime
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings

logger = logging.getLogger(__name__)

# Thread pool for async email sending
_executor = ThreadPoolExecutor(max_workers=2)


class EmailService:
    """
    Service for sending email alerts for stolen vehicle detections.
    Uses SMTP with TLS for secure email delivery.
    """
    
    def __init__(self):
        """Initialize email service with configuration from settings."""
        self.smtp_host = getattr(settings, 'SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.smtp_user = getattr(settings, 'SMTP_USER', '')
        self.smtp_password = getattr(settings, 'SMTP_PASSWORD', '')
        # FROM email - for services like Brevo that need a verified sender
        # Falls back to SMTP_USER if not set
        self.from_email = getattr(settings, 'SMTP_FROM_EMAIL', '') or self.smtp_user
        self.recipients = self._parse_recipients(
            getattr(settings, 'ALERT_EMAIL_RECIPIENTS', '')
        )
        self.enabled = getattr(settings, 'EMAIL_ALERTS_ENABLED', False)
        
        if self.enabled:
            logger.info(f"Email alerts enabled.")
            logger.info(f"  SMTP Host: {self.smtp_host}:{self.smtp_port}")
            logger.info(f"  From: {self.from_email}")
            logger.info(f"  Recipients: {self.recipients}")
        else:
            logger.info("Email alerts disabled. Set EMAIL_ALERTS_ENABLED=true to enable.")
    
    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """Parse comma-separated recipient string into list."""
        if not recipients_str:
            return []
        return [email.strip() for email in recipients_str.split(',') if email.strip()]
    
    def is_configured(self) -> bool:
        """Check if email service is properly configured."""
        return bool(
            self.enabled and 
            self.smtp_user and 
            self.smtp_password and 
            self.recipients
        )
    
    def send_stolen_vehicle_alert(
        self,
        plate_number: str,
        vehicle_type: str,
        vehicle_color: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        location: Optional[str] = None,
        image_base64: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> bool:
        """
        Send stolen vehicle alert email to configured recipients.
        
        Args:
            plate_number: Detected license plate number
            vehicle_type: Type of vehicle (car, truck, etc.)
            vehicle_color: Detected vehicle color
            confidence: Detection confidence score
            timestamp: Time of detection
            location: Detection location if available
            image_base64: Base64 encoded image of the detection
            source_type: Source of detection (camera, video, image)
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.warning("Email service not configured. Skipping alert.")
            return False
        
        try:
            timestamp = timestamp or datetime.now()
            
            # Create email message
            msg = MIMEMultipart('related')
            msg['Subject'] = f"🚨 STOLEN VEHICLE ALERT: {plate_number}"
            msg['From'] = self.from_email  # Use verified sender for Brevo
            msg['To'] = ', '.join(self.recipients)
            
            # Create HTML body
            html_body = self._create_alert_html(
                plate_number=plate_number,
                vehicle_type=vehicle_type,
                vehicle_color=vehicle_color,
                confidence=confidence,
                timestamp=timestamp,
                location=location,
                source_type=source_type,
                has_image=image_base64 is not None
            )
            
            # Attach HTML
            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)
            
            # Attach image if provided
            if image_base64:
                try:
                    # Remove data URL prefix if present
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    
                    image_data = base64.b64decode(image_base64)
                    image = MIMEImage(image_data, name='detection.jpg')
                    image.add_header('Content-ID', '<detection_image>')
                    msg.attach(image)
                except Exception as e:
                    logger.warning(f"Failed to attach image: {str(e)}")
            
            # Send email
            self._send_email(msg)
            
            logger.info(f"Stolen vehicle alert sent for plate: {plate_number}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send stolen vehicle alert: {str(e)}")
            return False
    
    def _send_email(self, msg: MIMEMultipart) -> None:
        """Send email via SMTP with TLS."""
        context = ssl.create_default_context()
        
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls(context=context)
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
    
    def send_alert_async(
        self,
        plate_number: str,
        vehicle_type: str,
        vehicle_color: Optional[str] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        location: Optional[str] = None,
        image_base64: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> None:
        """
        Send stolen vehicle alert asynchronously (non-blocking).
        Use this in the detection pipeline to avoid blocking.
        """
        if not self.is_configured():
            logger.debug("Email not configured, skipping async alert")
            return
        
        # Submit to thread pool
        _executor.submit(
            self.send_stolen_vehicle_alert,
            plate_number,
            vehicle_type,
            vehicle_color,
            confidence,
            timestamp,
            location,
            image_base64,
            source_type
        )
        logger.debug(f"Queued async email alert for plate: {plate_number}")
    
    def _create_alert_html(
        self,
        plate_number: str,
        vehicle_type: str,
        vehicle_color: Optional[str],
        confidence: Optional[float],
        timestamp: datetime,
        location: Optional[str],
        source_type: Optional[str],
        has_image: bool
    ) -> str:
        """Create HTML email body for stolen vehicle alert."""
        
        confidence_str = f"{confidence:.1%}" if confidence else "N/A"
        color_str = vehicle_color.capitalize() if vehicle_color else "Unknown"
        location_str = location or "Unknown"
        source_str = source_type.capitalize() if source_type else "Detection System"
        
        image_html = ""
        if has_image:
            image_html = '''
            <tr>
                <td colspan="2" style="padding: 20px; text-align: center;">
                    <img src="cid:detection_image" style="max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);" />
                    <p style="color: #666; font-size: 12px; margin-top: 8px;">Detection capture</p>
                </td>
            </tr>
            '''
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Arial, sans-serif; background-color: #f5f5f5;">
            <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 0 auto; background-color: #ffffff;">
                <!-- Header -->
                <tr>
                    <td style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 30px; text-align: center;">
                        <h1 style="color: #ffffff; margin: 0; font-size: 24px;">
                            🚨 STOLEN VEHICLE DETECTED
                        </h1>
                    </td>
                </tr>
                
                <!-- Alert Banner -->
                <tr>
                    <td style="background-color: #fff3cd; padding: 15px; text-align: center; border-bottom: 3px solid #ffc107;">
                        <span style="font-size: 28px; font-weight: bold; color: #856404; letter-spacing: 3px;">
                            {plate_number}
                        </span>
                    </td>
                </tr>
                
                <!-- Details -->
                <tr>
                    <td style="padding: 20px;">
                        <table width="100%" cellpadding="8" cellspacing="0" style="border-collapse: collapse;">
                            <tr>
                                <td style="font-weight: bold; color: #333; width: 40%;">Vehicle Type:</td>
                                <td style="color: #666;">{vehicle_type.capitalize()}</td>
                            </tr>
                            <tr style="background-color: #f9f9f9;">
                                <td style="font-weight: bold; color: #333;">Vehicle Color:</td>
                                <td style="color: #666;">
                                    <span style="display: inline-block; width: 12px; height: 12px; background-color: {self._get_color_code(vehicle_color)}; border-radius: 50%; margin-right: 8px; vertical-align: middle; border: 1px solid #ccc;"></span>
                                    {color_str}
                                </td>
                            </tr>
                            <tr>
                                <td style="font-weight: bold; color: #333;">Confidence:</td>
                                <td style="color: #666;">{confidence_str}</td>
                            </tr>
                            <tr style="background-color: #f9f9f9;">
                                <td style="font-weight: bold; color: #333;">Detection Time:</td>
                                <td style="color: #666;">{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                            </tr>
                            <tr>
                                <td style="font-weight: bold; color: #333;">Location:</td>
                                <td style="color: #666;">{location_str}</td>
                            </tr>
                            <tr style="background-color: #f9f9f9;">
                                <td style="font-weight: bold; color: #333;">Source:</td>
                                <td style="color: #666;">{source_str}</td>
                            </tr>
                            {image_html}
                        </table>
                    </td>
                </tr>
                
                <!-- Action Required -->
                <tr>
                    <td style="padding: 20px; background-color: #e8f4fd; border-top: 1px solid #b8daff;">
                        <p style="margin: 0; color: #004085; font-weight: bold;">
                            ⚡ Immediate Action Required
                        </p>
                        <p style="margin: 10px 0 0 0; color: #004085; font-size: 14px;">
                            Please verify this detection and take appropriate action. Log in to the system for full details.
                        </p>
                    </td>
                </tr>
                
                <!-- Footer -->
                <tr>
                    <td style="padding: 20px; background-color: #333; text-align: center;">
                        <p style="margin: 0; color: #999; font-size: 12px;">
                            Stolen Vehicle Detection System (SVDS)<br>
                            This is an automated alert. Do not reply to this email.
                        </p>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        '''
    
    def _get_color_code(self, color_name: Optional[str]) -> str:
        """Get hex color code for display."""
        color_map = {
            "red": "#FF0000",
            "orange": "#FFA500",
            "yellow": "#FFFF00",
            "green": "#008000",
            "blue": "#0000FF",
            "purple": "#800080",
            "pink": "#FFC0CB",
            "white": "#FFFFFF",
            "silver": "#C0C0C0",
            "gray": "#808080",
            "black": "#000000",
            "brown": "#8B4513",
            "beige": "#F5F5DC",
        }
        return color_map.get(color_name, "#CCCCCC") if color_name else "#CCCCCC"


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """
    Get or create singleton email service instance.
    
    Returns:
        EmailService: The email service instance
    """
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
