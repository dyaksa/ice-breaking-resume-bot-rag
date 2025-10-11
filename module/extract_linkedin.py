from typing import Optional, Dict, Any
import requests
import time
import logging

logger = logging.getLogger(__name__)


def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = time.time()

    try:
        if not api_key:
            raise ValueError("API key is required")

        api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"

        params = {
            "url": linkedin_profile_url,
            "fallback_to_cache": "on-error",
            "use_cache": "if-present",
            "skills": "include",
            "inferred_salary": "include",
            "personal_email": "include",
            "personal_contact_number": "include",
        }

        response = requests.post(
            api_endpoint, params=params, headers={"Authorization": f"Bearer {api_key}"}
        )

        logger.info(f"LinkedIn extraction took {time.time() - start_time:.2f} seconds")

        if response.status_code == 200:
            try:
                data = response.json()

                data = {
                    k: v
                    for k, v in data.items()
                    if v not in [[], "", None]
                    and k not in ["people_also_viewed", "certifications"]
                }

                if data.get("groups"):
                    for group_dict in data.get("groups"):
                        group_dict.pop("profile_pic_url", None)
                return data
            except Exception as e:
                logger.error(f"Error parsing JSON response: {e}")
                return {}

    except Exception as e:
        logger.error(f"Error extracting LinkedIn profile: {e}")
        return {}
