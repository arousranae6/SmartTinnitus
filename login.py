import hashlib
import hmac
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from backend.db import get_connection


PBKDF2_ITERATIONS = 310_000


# ============================================================
# PASSWORD HELPERS
# ============================================================

def hash_password(password, salt=None):

    if salt is None:
        salt_bytes = os.urandom(16)
    else:
        salt_bytes = bytes.fromhex(salt)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        str(password).encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS
    ).hex()

    return password_hash, salt_bytes.hex()


def verify_password(
    password,
    stored_hash,
    stored_salt
):

    calculated_hash, _ = hash_password(
        password,
        stored_salt
    )

    return hmac.compare_digest(
        calculated_hash,
        stored_hash
    )


# ============================================================
# USER HELPERS
# ============================================================

def username_exists(username):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE LOWER(username) = LOWER(?)
        LIMIT 1
        """,
        (
            str(username).strip(),
        )
    )

    result = cursor.fetchone()
    conn.close()

    return result is not None


def count_users(role=None):

    conn = get_connection()
    cursor = conn.cursor()

    if role is None:

        cursor.execute(
            "SELECT COUNT(*) FROM users"
        )

    else:

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM users
            WHERE role = ?
            """,
            (
                role,
            )
        )

    count = int(
        cursor.fetchone()[0]
    )

    conn.close()

    return count


# ============================================================
# CREATE ACCOUNT
# ============================================================

def create_user_account(
    username,
    password,
    role,
    patient_id=None,
    display_name=None
):

    username = str(
        username
    ).strip()

    password = str(
        password
    )

    role = str(
        role
    ).strip().lower()

    if not username:
        return False, "Username is required."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if role not in [
        "patient",
        "audiologist"
    ]:
        return False, "Invalid account role."

    if (
        role == "patient"
        and patient_id is None
    ):
        return False, "A patient account must be linked to a patient record."

    if username_exists(
        username
    ):
        return False, "This username already exists."

    password_hash, password_salt = (
        hash_password(
            password
        )
    )

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password_hash,
                password_salt,
                role,
                patient_id,
                display_name,
                is_active,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                password_hash,
                password_salt,
                role,
                patient_id,
                (
                    str(
                        display_name
                    ).strip()
                    if display_name
                    else None
                ),
                1,
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )
        )

        conn.commit()

    except Exception as error:

        conn.rollback()
        conn.close()

        return (
            False,
            f"Could not create account: {error}"
        )

    conn.close()

    return (
        True,
        "Account created successfully."
    )


# ============================================================
# LOGIN CHECK
# ============================================================

def authenticate_user(
    username,
    password,
    role
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            username,
            password_hash,
            password_salt,
            role,
            patient_id,
            display_name,
            is_active
        FROM users
        WHERE LOWER(username) = LOWER(?)
          AND role = ?
        LIMIT 1
        """,
        (
            str(
                username
            ).strip(),
            str(
                role
            ).strip().lower()
        )
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:

        return (
            False,
            None,
            "Incorrect username or password."
        )

    (
        user_id,
        stored_username,
        password_hash,
        password_salt,
        stored_role,
        patient_id,
        display_name,
        is_active
    ) = row

    if not bool(
        is_active
    ):

        return (
            False,
            None,
            "This account is inactive."
        )

    if not verify_password(
        password,
        password_hash,
        password_salt
    ):

        return (
            False,
            None,
            "Incorrect username or password."
        )

    user = {

        "user_id": user_id,

        "username": stored_username,

        "role": stored_role,

        "patient_id": patient_id,

        "display_name": (
            display_name
            or stored_username
        )
    }

    return (
        True,
        user,
        None
    )


# ============================================================
# SESSION STATE
# ============================================================

def initialize_auth_state():

    defaults = {

        "authenticated": False,

        "auth_user_id": None,

        "auth_username": None,

        "auth_role": None,

        "auth_patient_id": None,

        "auth_display_name": None,

        "login_selected_role": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[
                key
            ] = value


def set_authenticated_user(
    user
):

    st.session_state[
        "authenticated"
    ] = True

    st.session_state[
        "auth_user_id"
    ] = user[
        "user_id"
    ]

    st.session_state[
        "auth_username"
    ] = user[
        "username"
    ]

    st.session_state[
        "auth_role"
    ] = user[
        "role"
    ]

    st.session_state[
        "auth_patient_id"
    ] = user[
        "patient_id"
    ]

    st.session_state[
        "auth_display_name"
    ] = user[
        "display_name"
    ]


def logout_user():

    for key in [

        "authenticated",

        "auth_user_id",

        "auth_username",

        "auth_role",

        "auth_patient_id",

        "auth_display_name",

        "login_selected_role"

    ]:

        if key in st.session_state:

            del st.session_state[
                key
            ]

    initialize_auth_state()


# ============================================================
# FIRST AUDIOLOGIST
# ============================================================

def show_first_time_setup():

    st.html(
        """
<div class="setup-card">
    <div class="setup-eyebrow">
        FIRST-TIME SETUP
    </div>

    <div class="setup-title">
        Create the first audiologist account
    </div>

    <div class="setup-text">
        This account will manage patient access,
        create patient credentials and use the
        professional SmartTinnitus workspace.
    </div>
</div>
        """
    )

    with st.form(
        "first_audiologist_setup"
    ):

        col1, col2 = st.columns(
            2
        )

        with col1:

            display_name = st.text_input(
                "Audiologist name",
                placeholder="e.g. Dr. Sarah Ben Ali"
            )

            username = st.text_input(
                "Username",
                placeholder="Choose a username"
            )

        with col2:

            password = st.text_input(
                "Password",
                type="password",
                placeholder="Minimum 8 characters"
            )

            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Repeat password"
            )

        submitted = st.form_submit_button(
            "Create Audiologist Account",
            use_container_width=True
        )

    if submitted:

        if (
            password
            != confirm_password
        ):

            st.error(
                "The two passwords do not match."
            )

            return

        success, message = (
            create_user_account(
                username=username,
                password=password,
                role="audiologist",
                patient_id=None,
                display_name=display_name
            )
        )

        if success:

            st.success(
                "Audiologist account created successfully."
            )

            st.rerun()

        else:

            st.error(
                message
            )


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

    initialize_auth_state()

    app_dir = Path(__file__).resolve().parent
    logo_path = app_dir / "assets" / "medico_logo.png"

    if "login_show_contact" not in st.session_state:
        st.session_state["login_show_contact"] = False

    st.html(
        """
<style>
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #edf0ee !important;
}
[data-testid="stToolbar"] { visibility: hidden; }
.stApp { background: #f7f8f6; }
.block-container {
    max-width: 980px !important;
    padding-top: 3.2rem !important;
    padding-bottom: 2.5rem !important;
}

/* Login brand */
.login-brand-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.55rem;
    margin: 0 auto 1.7rem auto;
    padding: .35rem 0 .2rem 0;
}
.login-brand-text {
    text-align: left;
    max-width: 610px;
}
.login-brand-kicker {
    color: #008f5a;
    font-size: .72rem;
    font-weight: 850;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .18rem;
}
.login-brand-title {
    color: #18231e;
    font-size: 2.15rem;
    line-height: 1.08;
    font-weight: 850;
    letter-spacing: -.035em;
    margin: 0;
}
.login-brand-copy {
    max-width: 610px;
    margin: .5rem 0 0 0;
    color: #69766f;
    font-size: .95rem;
    line-height: 1.55;
}
@media (max-width: 760px) {
    .login-brand-row {
        flex-direction: column;
        gap: .8rem;
        text-align: center;
    }
    .login-brand-text {
        text-align: center;
    }
    .login-brand-copy {
        margin-left: auto;
        margin-right: auto;
    }
}

/* Main access panel */
.login-panel {
    background: #ffffff;
    border: 1px solid #e2e8e4;
    border-radius: 18px;
    padding: 1.45rem 1.5rem 1.25rem 1.5rem;
    box-shadow: 0 10px 34px rgba(22,63,47,.055);
    margin: 0 auto 1.1rem auto;
}
.login-panel-kicker {
    color: #008f5a;
    font-size: .7rem;
    font-weight: 850;
    letter-spacing: .11em;
    text-transform: uppercase;
    margin-bottom: .2rem;
}
.login-panel-title {
    color: #18382e;
    font-size: 1.35rem;
    font-weight: 820;
    margin-bottom: .28rem;
}
.login-panel-copy {
    color: #75817b;
    font-size: .88rem;
    line-height: 1.5;
}
.role-card-copy {
    min-height: 78px;
    color: #66746d;
    font-size: .87rem;
    line-height: 1.5;
}
.role-card-title {
    color: #153d31;
    font-size: 1.08rem;
    font-weight: 820;
    margin-bottom: .35rem;
}
.role-icon {
    font-size: 1.65rem;
    margin-bottom: .45rem;
}

/* Streamlit containers/cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #dfe7e2 !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 16px rgba(19,64,46,.035) !important;
}
.stButton > button,
.stFormSubmitButton > button {
    min-height: 44px !important;
    border-radius: 8px !important;
    font-weight: 750 !important;
    border: 1px solid #008f5a !important;
    background: #008f5a !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #006f47 !important;
    border-color: #006f47 !important;
    color: #ffffff !important;
}
div[data-baseweb="input"] > div {
    border-radius: 8px !important;
    border-color: #dfe6e2 !important;
}
.login-note {
    background: #f1f8f4;
    border-left: 4px solid #008f5a;
    border-radius: 8px;
    padding: .8rem .95rem;
    color: #596961;
    font-size: .84rem;
    line-height: 1.5;
    margin-top: .8rem;
}
.login-footer {
    text-align: center;
    color: #929c97;
    font-size: .76rem;
    margin-top: 1.35rem;
}
.contact-box {
    background:#ffffff;
    border:1px solid #e2e8e4;
    border-left:4px solid #008f5a;
    border-radius:10px;
    padding:13px 16px;
    margin-top:.55rem;
    color:#55645d;
    font-size:.84rem;
    line-height:1.65;
}
</style>
        """
    )

    # --------------------------------------------------------
    # MEDICO LOGO + SMARTTINNITUS BRAND ON ONE ROW
    # --------------------------------------------------------
    brand_logo_col, brand_text_col = st.columns(
        [1.35, 2.65],
        gap="medium",
        vertical_alignment="center"
    )

    with brand_logo_col:
        if logo_path.exists():
            st.image(str(logo_path), width=165)

    with brand_text_col:
        st.html(
            """
<div class="login-brand-text">
    <div class="login-brand-title">SmartTinnitus</div>
    <div class="login-brand-copy">
        Clinician-supervised digital audiology platform for structured tinnitus assessment,
        care coordination and personalized acoustic testing.
    </div>
</div>
            """
        )

    st.write("")

    selected_role = st.session_state.get("login_selected_role")

    # ========================================================
    # ROLE SELECTION
    # ========================================================
    if selected_role is None:

        st.html(
            """
<div class="login-panel">
    <div class="login-panel-kicker">Secure access</div>
    <div class="login-panel-title">Choose your workspace</div>
    <div class="login-panel-copy">
        Select the portal that matches your role. You can return here at any time before signing in.
    </div>
</div>
            """
        )

        patient_col, audiologist_col = st.columns(2, gap="large")

        with patient_col:
            with st.container(border=True):
                st.html(
                    """
<div class="role-icon">👤</div>
<div class="role-card-title">Patient Portal</div>
<div class="role-card-copy">
    Review your tinnitus assessment, profile, appointments and care requests,
    and communicate through the SmartTinnitus Assistant.
</div>
                    """
                )
                if st.button(
                    "Enter Patient Portal",
                    use_container_width=True,
                    key="patient_access_button"
                ):
                    st.session_state["login_selected_role"] = "patient"
                    st.rerun()

        with audiologist_col:
            with st.container(border=True):
                st.html(
                    """
<div class="role-icon">🩺</div>
<div class="role-card-title">Audiologist Portal</div>
<div class="role-card-copy">
    Manage patients, coordinate care, review audiometry, supervise acoustic personalization,
    use the Digital Twin and create clinical reports.
</div>
                    """
                )
                if st.button(
                    "Enter Audiologist Portal",
                    use_container_width=True,
                    key="audiologist_access_button"
                ):
                    st.session_state["login_selected_role"] = "audiologist"
                    st.rerun()

        st.html(
            """
<div class="login-note">
    <strong>Patient accounts are created by the audiologist.</strong>
    Patients receive their SmartTinnitus username and temporary password from the clinic.
</div>
            """
        )

        if count_users("audiologist") == 0:
            st.markdown("---")
            show_first_time_setup()

    # ========================================================
    # PATIENT LOGIN
    # ========================================================
    elif selected_role == "patient":

        left, center, right = st.columns([1, 1.55, 1])
        with center:
            with st.container(border=True):
                st.html(
                    """
<div class="login-panel-kicker">Patient portal</div>
<div class="login-panel-title">Sign in to SmartTinnitus</div>
<div class="login-panel-copy">
    Use the credentials provided by your audiologist.
</div>
                    """
                )

                with st.form("patient_login_form"):
                    username = st.text_input("Username", placeholder="Example: ST-P001")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    login_clicked = st.form_submit_button("Sign In", use_container_width=True)

                if login_clicked:
                    success, user, error = authenticate_user(
                        username=username,
                        password=password,
                        role="patient"
                    )
                    if success:
                        set_authenticated_user(user)
                        st.rerun()
                    else:
                        st.error(error)

                if st.button("← Back to portal choice", use_container_width=True, key="patient_login_back"):
                    st.session_state["login_selected_role"] = None
                    st.rerun()

    # ========================================================
    # AUDIOLOGIST LOGIN
    # ========================================================
    elif selected_role == "audiologist":

        left, center, right = st.columns([1, 1.55, 1])
        with center:
            with st.container(border=True):
                st.html(
                    """
<div class="login-panel-kicker">Professional portal</div>
<div class="login-panel-title">Audiologist sign in</div>
<div class="login-panel-copy">
    Access the clinician-supervised SmartTinnitus workspace.
</div>
                    """
                )

                if count_users("audiologist") == 0:
                    st.warning("No audiologist account exists yet. Create the first account below.")
                    show_first_time_setup()
                else:
                    with st.form("audiologist_login_form"):
                        username = st.text_input("Username", placeholder="Audiologist username")
                        password = st.text_input("Password", type="password", placeholder="Enter your password")
                        login_clicked = st.form_submit_button("Sign In", use_container_width=True)

                    if login_clicked:
                        success, user, error = authenticate_user(
                            username=username,
                            password=password,
                            role="audiologist"
                        )
                        if success:
                            set_authenticated_user(user)
                            st.rerun()
                        else:
                            st.error(error)

                if st.button("← Back to portal choice", use_container_width=True, key="audiologist_login_back"):
                    st.session_state["login_selected_role"] = None
                    st.rerun()

    # --------------------------------------------------------
    # CONTACT: HIDDEN BY DEFAULT
    # --------------------------------------------------------
    st.markdown("")
    contact_left, contact_mid, contact_right = st.columns([1.5, .8, 1.5])
    with contact_mid:
        contact_label = "Close contact" if st.session_state["login_show_contact"] else "Contact"
        if st.button(contact_label, use_container_width=True, key="login_contact_toggle"):
            st.session_state["login_show_contact"] = not st.session_state["login_show_contact"]
            st.rerun()

    if st.session_state["login_show_contact"]:
        st.html(
            """
<div class="contact-box">
    <strong>Medico International Services</strong><br>
    12 Av Slimene Ben Slimene – Diar el Ons Bloc S – 2éme étage, 2092 El Manar 2<br>
    Tel: 98 119 359<br>
    Email: rendezvous@medico-internationalservices.com.tn<br>
    <a href="https://medico-internationalservices.com.tn/" target="_blank" rel="noopener noreferrer"
       style="color:#008f5a;font-weight:750;text-decoration:none;">
        Visit official Medico International Services website ↗
    </a>
</div>
            """
        )

    st.html(
        """
<div class="login-footer">
    © ABC 2023 · SmartTinnitus research and engineering prototype
</div>
        """
    )


# ============================================================
# SIDEBAR USER CARD
# ============================================================

def show_logged_in_user_card():

    initialize_auth_state()

    role = (
        st.session_state[
            "auth_role"
        ]
    )

    display_name = (
        st.session_state[
            "auth_display_name"
        ]
        or
        st.session_state[
            "auth_username"
        ]
    )

    st.markdown(
        f"**{display_name}**"
    )

    st.caption(
        (
            "Patient"
            if role == "patient"
            else "Audiologist"
        )
    )

    if st.button(
        "Log out",
        use_container_width=True
    ):

        logout_user()

        st.rerun()
