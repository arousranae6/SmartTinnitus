import secrets
import string

from login import create_user_account


# ============================================================
# PATIENT USERNAME GENERATOR
# ============================================================

def generate_patient_username(patient_code):
    """
    Generate a username using the patient code.

    Example:
        P001 -> ST-P001
    """

    patient_code = str(patient_code).strip().upper()

    return f"ST-{patient_code}"


# ============================================================
# TEMPORARY PASSWORD GENERATOR
# ============================================================

def generate_temporary_password(length=12):
    """
    Generate a random temporary password.
    """

    uppercase = string.ascii_uppercase
    lowercase = string.ascii_lowercase
    digits = string.digits
    symbols = "!@#$%&*"

    # Make sure password contains:
    # 1 uppercase
    # 1 lowercase
    # 1 number
    # 1 symbol

    password_characters = [
        secrets.choice(uppercase),
        secrets.choice(lowercase),
        secrets.choice(digits),
        secrets.choice(symbols)
    ]

    all_characters = (
        uppercase
        + lowercase
        + digits
        + symbols
    )

    # Fill remaining characters
    for _ in range(length - 4):

        password_characters.append(
            secrets.choice(
                all_characters
            )
        )

    # Shuffle characters
    secrets.SystemRandom().shuffle(
        password_characters
    )

    return "".join(
        password_characters
    )


# ============================================================
# CREATE PATIENT LOGIN
# ============================================================

def create_patient_login(
    patient_id,
    patient_code
):
    """
    Create a SmartTinnitus patient account.

    The account is linked directly to the patient database ID.
    """

    username = generate_patient_username(
        patient_code
    )

    temporary_password = (
        generate_temporary_password()
    )

    success, message = create_user_account(
        username=username,
        password=temporary_password,
        role="patient",
        patient_id=patient_id,
        display_name=patient_code
    )

    return (
        success,
        message,
        username,
        temporary_password
    )