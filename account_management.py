import pandas as pd
import streamlit as st
from datetime import datetime

from database import get_connection
from login import (
    create_user_account,
    hash_password
)


# ============================================================
# HELPERS
# ============================================================


def _load_accounts():
    conn = get_connection()

    accounts_df = pd.read_sql_query(
        """
        SELECT
            users.id,
            users.username,
            users.display_name,
            users.role,
            users.patient_id,
            patients.patient_code AS patient_code,
            users.is_active,
            users.created_at
        FROM users
        LEFT JOIN patients
            ON patients.id = users.patient_id
        ORDER BY users.id ASC
        """,
        conn
    )

    conn.close()

    return accounts_df


def _load_patients():
    conn = get_connection()

    patients_df = pd.read_sql_query(
        """
        SELECT
            id,
            patient_code,
            age,
            sex
        FROM patients
        ORDER BY patient_code ASC, id ASC
        """,
        conn
    )

    conn.close()

    return patients_df


def _patient_account_already_linked(
    patient_id,
    exclude_user_id=None
):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT id
    FROM users
    WHERE role = 'patient'
      AND patient_id = ?
    """

    params = [patient_id]

    if exclude_user_id is not None:
        query += " AND id != ?"
        params.append(exclude_user_id)

    query += " LIMIT 1"

    cursor.execute(
        query,
        tuple(params)
    )

    row = cursor.fetchone()
    conn.close()

    return row is not None


def _username_taken(
    username,
    exclude_user_id=None
):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT id
    FROM users
    WHERE LOWER(username) = LOWER(?)
    """

    params = [
        str(username).strip()
    ]

    if exclude_user_id is not None:
        query += " AND id != ?"
        params.append(exclude_user_id)

    query += " LIMIT 1"

    cursor.execute(
        query,
        tuple(params)
    )

    row = cursor.fetchone()
    conn.close()

    return row is not None


def _update_account_details(
    user_id,
    username,
    display_name,
    patient_id
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET
            username = ?,
            display_name = ?,
            patient_id = ?
        WHERE id = ?
        """,
        (
            str(username).strip(),
            (
                str(display_name).strip()
                if display_name
                else None
            ),
            patient_id,
            user_id
        )
    )

    conn.commit()
    conn.close()


def _set_account_status(
    user_id,
    is_active
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET is_active = ?
        WHERE id = ?
        """,
        (
            1 if is_active else 0,
            user_id
        )
    )

    conn.commit()
    conn.close()


def _reset_account_password(
    user_id,
    new_password
):
    password_hash, password_salt = (
        hash_password(
            new_password
        )
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE users
        SET
            password_hash = ?,
            password_salt = ?
        WHERE id = ?
        """,
        (
            password_hash,
            password_salt,
            user_id
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# MAIN PAGE
# ============================================================


def show_account_management():

    st.title("User Account Management")

    st.caption(
        "Create, review and manage SmartTinnitus login accounts. "
        "This page is available only to audiologists."
    )

    # ========================================================
    # ACCOUNT OVERVIEW
    # ========================================================

    accounts_df = _load_accounts()

    st.markdown("## Account Overview")

    if accounts_df.empty:

        st.info(
            "No login accounts have been created yet."
        )

    else:

        display_df = accounts_df.copy()

        display_df["Role"] = (
            display_df["role"]
            .str.title()
        )

        display_df["Linked Patient"] = (
            display_df["patient_code"]
            .fillna("—")
        )

        display_df["Status"] = (
            display_df["is_active"]
            .map(
                {
                    1: "Active",
                    0: "Inactive"
                }
            )
            .fillna("Unknown")
        )

        display_df = display_df[
            [
                "id",
                "username",
                "display_name",
                "Role",
                "Linked Patient",
                "Status",
                "created_at"
            ]
        ]

        display_df.columns = [
            "Account ID",
            "Username",
            "Display Name",
            "Role",
            "Linked Patient",
            "Status",
            "Created"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # ========================================================
    # CREATE ACCOUNTS
    # ========================================================

    st.markdown("## Create New Account")

    account_type = st.radio(
        "Account type",
        [
            "Patient",
            "Audiologist"
        ],
        horizontal=True,
        key="create_account_type"
    )

    # --------------------------------------------------------
    # CREATE PATIENT ACCOUNT
    # --------------------------------------------------------

    if account_type == "Patient":

        patients_df = _load_patients()

        if patients_df.empty:

            st.warning(
                "Create a patient record first from the Patients page "
                "before creating a patient login."
            )

        else:

            patient_options = {
                f"{row['patient_code']} (ID {int(row['id'])})":
                    int(row["id"])
                for _, row in patients_df.iterrows()
            }

            with st.form(
                "create_patient_login_form"
            ):

                selected_patient_label = (
                    st.selectbox(
                        "Link account to patient",
                        list(
                            patient_options.keys()
                        )
                    )
                )

                selected_patient_id = (
                    patient_options[
                        selected_patient_label
                    ]
                )

                default_display_name = (
                    selected_patient_label.split(
                        " (ID "
                    )[0]
                )

                display_name = st.text_input(
                    "Display name",
                    value=default_display_name
                )

                username = st.text_input(
                    "Patient username"
                )

                password = st.text_input(
                    "Temporary password",
                    type="password"
                )

                confirm_password = st.text_input(
                    "Confirm temporary password",
                    type="password"
                )

                create_clicked = (
                    st.form_submit_button(
                        "Create Patient Account",
                        use_container_width=True
                    )
                )

            if create_clicked:

                if password != confirm_password:

                    st.error(
                        "The two passwords do not match."
                    )

                elif _patient_account_already_linked(
                    selected_patient_id
                ):

                    st.error(
                        "This patient record is already linked to a patient login account."
                    )

                else:

                    success, message = (
                        create_user_account(
                            username=username,
                            password=password,
                            role="patient",
                            patient_id=selected_patient_id,
                            display_name=display_name
                        )
                    )

                    if success:

                        st.success(
                            "Patient login created successfully."
                        )

                        st.rerun()

                    else:

                        st.error(
                            message
                        )

            st.caption(
                "Each patient login is linked to one patient record so that "
                "the patient can only access their own SmartTinnitus data."
            )

    # --------------------------------------------------------
    # CREATE AUDIOLOGIST ACCOUNT
    # --------------------------------------------------------

    else:

        with st.form(
            "create_audiologist_login_form"
        ):

            display_name = st.text_input(
                "Audiologist display name"
            )

            username = st.text_input(
                "Audiologist username"
            )

            password = st.text_input(
                "Temporary password",
                type="password"
            )

            confirm_password = st.text_input(
                "Confirm temporary password",
                type="password"
            )

            create_clicked = (
                st.form_submit_button(
                    "Create Audiologist Account",
                    use_container_width=True
                )
            )

        if create_clicked:

            if password != confirm_password:

                st.error(
                    "The two passwords do not match."
                )

            else:

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

    st.markdown("---")

    # ========================================================
    # MANAGE EXISTING ACCOUNT
    # ========================================================

    st.markdown("## Manage Existing Account")

    accounts_df = _load_accounts()

    if accounts_df.empty:

        st.info(
            "There are no accounts available to manage."
        )

        return

    account_labels = {}

    for _, row in accounts_df.iterrows():

        role_label = str(
            row["role"]
        ).title()

        status_label = (
            "Active"
            if int(
                row["is_active"]
            ) == 1
            else "Inactive"
        )

        label = (
            f"{row['username']} — {role_label} — {status_label}"
        )

        account_labels[
            label
        ] = int(
            row["id"]
        )

    selected_account_label = (
        st.selectbox(
            "Select account",
            list(
                account_labels.keys()
            ),
            key="manage_account_selector"
        )
    )

    selected_user_id = (
        account_labels[
            selected_account_label
        ]
    )

    selected_row = (
        accounts_df[
            accounts_df[
                "id"
            ]
            ==
            selected_user_id
        ]
        .iloc[
            0
        ]
    )

    selected_role = str(
        selected_row[
            "role"
        ]
    )

    selected_is_active = bool(
        int(
            selected_row[
                "is_active"
            ]
        )
    )

    st.markdown(
        f"### {str(selected_row['username'])}"
    )

    c1, c2, c3 = st.columns(
        3
    )

    c1.metric(
        "Role",
        selected_role.title()
    )

    c2.metric(
        "Status",
        (
            "Active"
            if selected_is_active
            else "Inactive"
        )
    )

    c3.metric(
        "Linked Patient",
        (
            str(
                selected_row[
                    "patient_code"
                ]
            )
            if pd.notna(
                selected_row[
                    "patient_code"
                ]
            )
            else "—"
        )
    )

    # --------------------------------------------------------
    # EDIT ACCOUNT DETAILS
    # --------------------------------------------------------

    st.markdown("#### Edit Account Details")

    patients_df = _load_patients()

    with st.form(
        "edit_account_details_form"
    ):

        edited_username = st.text_input(
            "Username",
            value=str(
                selected_row[
                    "username"
                ]
            )
        )

        edited_display_name = st.text_input(
            "Display name",
            value=(
                str(
                    selected_row[
                        "display_name"
                    ]
                )
                if pd.notna(
                    selected_row[
                        "display_name"
                    ]
                )
                else ""
            )
        )

        edited_patient_id = None

        if (
            selected_role
            ==
            "patient"
        ):

            patient_options = {
                f"{row['patient_code']} (ID {int(row['id'])})":
                    int(row["id"])
                for _, row in patients_df.iterrows()
            }

            option_labels = list(
                patient_options.keys()
            )

            current_patient_id = (
                int(
                    selected_row[
                        "patient_id"
                    ]
                )
                if pd.notna(
                    selected_row[
                        "patient_id"
                    ]
                )
                else None
            )

            current_index = 0

            for index, label in enumerate(
                option_labels
            ):

                if (
                    patient_options[
                        label
                    ]
                    ==
                    current_patient_id
                ):

                    current_index = (
                        index
                    )

                    break

            selected_patient_label = (
                st.selectbox(
                    "Linked patient",
                    option_labels,
                    index=current_index
                )
            )

            edited_patient_id = (
                patient_options[
                    selected_patient_label
                ]
            )

        st.caption(
            "The account role is fixed. Create a new account if a different role is required."
        )

        save_details_clicked = (
            st.form_submit_button(
                "Save Account Changes",
                use_container_width=True
            )
        )

    if save_details_clicked:

        if len(
            str(
                edited_username
            ).strip()
        ) < 3:

            st.error(
                "Username must contain at least 3 characters."
            )

        elif _username_taken(
            edited_username,
            exclude_user_id=selected_user_id
        ):

            st.error(
                "This username is already used by another account."
            )

        elif (
            selected_role
            ==
            "patient"
            and
            _patient_account_already_linked(
                edited_patient_id,
                exclude_user_id=selected_user_id
            )
        ):

            st.error(
                "The selected patient record is already linked to another patient account."
            )

        else:

            _update_account_details(
                user_id=selected_user_id,
                username=edited_username,
                display_name=edited_display_name,
                patient_id=(
                    edited_patient_id
                    if selected_role
                    ==
                    "patient"
                    else None
                )
            )

            # Keep current-session display information synchronized
            # when the audiologist edits their own account.
            if (
                selected_user_id
                ==
                st.session_state.get(
                    "auth_user_id"
                )
            ):

                st.session_state[
                    "auth_username"
                ] = str(
                    edited_username
                ).strip()

                st.session_state[
                    "auth_display_name"
                ] = (
                    str(
                        edited_display_name
                    ).strip()
                    if edited_display_name
                    else str(
                        edited_username
                    ).strip()
                )

            st.success(
                "Account details updated successfully."
            )

            st.rerun()

    # --------------------------------------------------------
    # ACTIVATE / DEACTIVATE
    # --------------------------------------------------------

    st.markdown("#### Account Status")

    current_logged_in_user_id = (
        st.session_state.get(
            "auth_user_id"
        )
    )

    if (
        selected_user_id
        ==
        current_logged_in_user_id
    ):

        st.info(
            "You cannot deactivate the account that is currently logged in."
        )

    else:

        if selected_is_active:

            if st.button(
                "Deactivate Account",
                use_container_width=True,
                key="deactivate_account_button"
            ):

                _set_account_status(
                    selected_user_id,
                    False
                )

                st.success(
                    "Account deactivated. The user can no longer sign in."
                )

                st.rerun()

        else:

            if st.button(
                "Reactivate Account",
                use_container_width=True,
                key="reactivate_account_button"
            ):

                _set_account_status(
                    selected_user_id,
                    True
                )

                st.success(
                    "Account reactivated."
                )

                st.rerun()

    # --------------------------------------------------------
    # RESET PASSWORD
    # --------------------------------------------------------

    st.markdown("#### Reset Password")

    st.caption(
        "Password reset replaces the existing password with a new hashed password."
    )

    with st.form(
        "reset_account_password_form"
    ):

        new_password = st.text_input(
            "New temporary password",
            type="password"
        )

        confirm_new_password = st.text_input(
            "Confirm new temporary password",
            type="password"
        )

        reset_password_clicked = (
            st.form_submit_button(
                "Reset Password",
                use_container_width=True
            )
        )

    if reset_password_clicked:

        if (
            new_password
            !=
            confirm_new_password
        ):

            st.error(
                "The two passwords do not match."
            )

        elif len(
            new_password
        ) < 8:

            st.error(
                "Password must contain at least 8 characters."
            )

        else:

            _reset_account_password(
                selected_user_id,
                new_password
            )

            st.success(
                "Password reset successfully."
            )

    st.markdown("---")

    st.caption(
        "Accounts are deactivated rather than deleted so that the prototype "
        "retains account-history consistency. SmartTinnitus authentication "
        "remains an academic prototype and is not a production clinical "
        "identity-management system."
    )
