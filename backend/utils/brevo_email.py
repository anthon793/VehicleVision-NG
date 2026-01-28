"""
Brevo (Sendinblue) Transactional Email Utility.

Uses Brevo's API with email templates for reliable, professional email delivery.
This is preferred over SMTP for production use.

Usage:
    from utils.brevo_email import send_missing_vehicle_email
    
    await send_missing_vehicle_email(
        user_email="admin@example.com",
        user_name="Admin",
        vehicle_details={
            "plate_number": "ABC123XY",
            "vehicle_color": "Black",
            "vehicle_type": "Car",
            "location_found": "Lagos, Nigeria",
            "date_time_found": "2024-01-26 15:30:00"
        }
    )
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Brevo SDK
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Thread pool for async email sending
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="brevo_email")


class BrevoEmailService:
    """
    Production-ready Brevo transactional email service.
    
    Uses Brevo's API with email templates for:
    - Reliable delivery
    - Professional HTML templates
    - Tracking and analytics
    - No SMTP connection issues
    """
    
    def __init__(self):
        """Initialize Brevo API client."""
        self.api_key = os.getenv("BREVO_API_KEY", "")
        self.default_from_email = os.getenv("DEFAULT_FROM_EMAIL", "noreply@example.com")
        self.default_from_name = os.getenv("DEFAULT_FROM_NAME", "Vehicle Detection System")
        self.template_id = int(os.getenv("BREVO_TEMPLATE_ID", "1"))
        self.enabled = bool(self.api_key)
        
        # Configure Brevo API
        self.configuration = sib_api_v3_sdk.Configuration()
        self.configuration.api_key['api-key'] = self.api_key
        
        if self.enabled:
            logger.info("Brevo Email Service initialized")
            logger.info(f"  From: {self.default_from_name} <{self.default_from_email}>")
            logger.info(f"  Template ID: {self.template_id}")
        else:
            logger.warning("Brevo Email Service disabled - BREVO_API_KEY not set")
    
    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        return self.enabled and bool(self.api_key)
    
    def send_missing_vehicle_email(
        self,
        user_email: str,
        user_name: str,
        vehicle_details: Dict[str, Any],
        template_id: Optional[int] = None
    ) -> bool:
        """
        Send a missing/stolen vehicle found notification email.
        
        Args:
            user_email: Recipient email address
            user_name: Recipient name (for personalization)
            vehicle_details: Dictionary containing:
                - plate_number: str
                - vehicle_color: str
                - vehicle_type: str
                - location_found: str
                - date_time_found: str
            template_id: Optional Brevo template ID (uses default if not provided)
        
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        if not self.is_configured():
            logger.warning("Brevo not configured. Skipping email.")
            return False
        
        try:
            # Create API instance
            api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
                sib_api_v3_sdk.ApiClient(self.configuration)
            )
            
            # Prepare template parameters
            params = {
                "name": user_name,
                "plate_number": vehicle_details.get("plate_number", "Unknown"),
                "vehicle_color": vehicle_details.get("vehicle_color", "Unknown"),
                "vehicle_type": vehicle_details.get("vehicle_type", "Vehicle"),
                "location_found": vehicle_details.get("location_found", "Unknown Location"),
                "date_time_found": vehicle_details.get("date_time_found", 
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                # Additional params you might want
                "confidence": vehicle_details.get("confidence", "N/A"),
                "source": vehicle_details.get("source", "Detection System"),
            }
            
            # Create email request
            send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
                to=[{"email": user_email, "name": user_name}],
                template_id=template_id or self.template_id,
                params=params,
                sender={
                    "email": self.default_from_email,
                    "name": self.default_from_name
                },
                # Optional: Add reply-to
                reply_to={
                    "email": self.default_from_email,
                    "name": self.default_from_name
                },
                # Optional: Add headers for tracking
                headers={
                    "X-Vehicle-Plate": vehicle_details.get("plate_number", ""),
                    "X-Detection-Time": vehicle_details.get("date_time_found", "")
                }
            )
            
            # Send the email
            response = api_instance.send_transac_email(send_smtp_email)
            
            logger.info(f"Stolen vehicle alert sent to {user_email}")
            logger.info(f"  Plate: {vehicle_details.get('plate_number')}")
            logger.info(f"  Message ID: {response.message_id}")
            
            return True
            
        except ApiException as e:
            logger.error(f"Brevo API error: {e.status} - {e.reason}")
            logger.error(f"  Body: {e.body}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def send_email_async(
        self,
        user_email: str,
        user_name: str,
        vehicle_details: Dict[str, Any],
        template_id: Optional[int] = None
    ) -> None:
        """
        Send email asynchronously (non-blocking).
        
        Use this in API endpoints to avoid blocking the response.
        """
        if not self.is_configured():
            logger.debug("Brevo not configured. Skipping async email.")
            return
        
        # Submit to thread pool
        _executor.submit(
            self.send_missing_vehicle_email,
            user_email,
            user_name,
            vehicle_details,
            template_id
        )
        logger.debug(f"Queued async email for plate: {vehicle_details.get('plate_number')}")


# Singleton instance
_brevo_service: Optional[BrevoEmailService] = None


def get_brevo_service() -> BrevoEmailService:
    """Get or create singleton Brevo service instance."""
    global _brevo_service
    if _brevo_service is None:
        _brevo_service = BrevoEmailService()
    return _brevo_service


# Convenience function for direct use
def send_missing_vehicle_email(
    user_email: str,
    user_name: str,
    vehicle_details: Dict[str, Any],
    template_id: Optional[int] = None,
    async_send: bool = True
) -> bool:
    """
    Send a missing/stolen vehicle found notification email.
    
    This is the main function to call from your endpoints.
    
    Args:
        user_email: Recipient email address
        user_name: Recipient name
        vehicle_details: Dict with plate_number, vehicle_color, vehicle_type, 
                        location_found, date_time_found
        template_id: Optional Brevo template ID
        async_send: If True, sends asynchronously (non-blocking)
    
    Returns:
        bool: True if sent/queued successfully
    
    Example:
        send_missing_vehicle_email(
            user_email="admin@example.com",
            user_name="Admin",
            vehicle_details={
                "plate_number": "ABC123XY",
                "vehicle_color": "Black",
                "vehicle_type": "Car",
                "location_found": "Lagos, Nigeria",
                "date_time_found": "2024-01-26 15:30:00"
            }
        )
    """
    service = get_brevo_service()
    
    if async_send:
        service.send_email_async(user_email, user_name, vehicle_details, template_id)
        return True
    else:
        return service.send_missing_vehicle_email(user_email, user_name, vehicle_details, template_id)


# Async coroutine version for use with await
async def send_missing_vehicle_email_async(
    user_email: str,
    user_name: str,
    vehicle_details: Dict[str, Any],
    template_id: Optional[int] = None
) -> bool:
    """
    Async version - await this in async endpoints.
    
    Example:
        result = await send_missing_vehicle_email_async(
            user_email="admin@example.com",
            user_name="Admin",
            vehicle_details={...}
        )
    """
    service = get_brevo_service()
    loop = asyncio.get_event_loop()
    
    return await loop.run_in_executor(
        _executor,
        service.send_missing_vehicle_email,
        user_email,
        user_name,
        vehicle_details,
        template_id
    )
