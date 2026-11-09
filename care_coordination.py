import streamlit as st
import pandas as pd
from datetime import datetime, date, time
from database import get_connection


def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def ensure_care_coordination_table(connection=None):
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS care_requests
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                direction TEXT NOT NULL,
                request_type TEXT NOT NULL,
                communication_mode TEXT,
                specialist TEXT,
                title TEXT,
                message TEXT,
                preferred_datetime TEXT,
                scheduled_datetime TEXT,
                status TEXT NOT NULL DEFAULT 'Pending',
                patient_response TEXT,
                audiologist_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        if owns_connection:
            conn.close()


def _insert_request(
    conn,
    patient_id,
    direction,
    request_type,
    title,
    message,
    communication_mode=None,
    specialist=None,
    preferred_datetime=None,
    scheduled_datetime=None,
    status='Pending',
):
    conn.execute(
        """
        INSERT INTO care_requests
        (
            patient_id,
            direction,
            request_type,
            communication_mode,
            specialist,
            title,
            message,
            preferred_datetime,
            scheduled_datetime,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            direction,
            request_type,
            communication_mode,
            specialist,
            title,
            message,
            preferred_datetime,
            scheduled_datetime,
            status,
            _now(),
            _now(),
        ),
    )
    conn.commit()


def get_patient_care_summary(patient_id):
    if patient_id is None:
        return {'pending': 0, 'scheduled': 0, 'action_required': 0, 'latest': None}
    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        pending = conn.execute(
            """
            SELECT COUNT(*)
            FROM care_requests
            WHERE patient_id=?
              AND status NOT IN ('Completed','Cancelled')
            """,
            (patient_id,),
        ).fetchone()[0]
        scheduled = conn.execute(
            "SELECT COUNT(*) FROM care_requests WHERE patient_id=? AND status IN ('Scheduled','Accepted')",
            (patient_id,),
        ).fetchone()[0]
        action_required = conn.execute(
            """
            SELECT COUNT(*)
            FROM care_requests
            WHERE patient_id=?
              AND direction='audiologist_to_patient'
              AND status IN ('Pending','Needs Information','Needs Confirmation','Referred')
            """,
            (patient_id,),
        ).fetchone()[0]
        latest = conn.execute(
            """
            SELECT request_type, title, status, scheduled_datetime, created_at
            FROM care_requests
            WHERE patient_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (patient_id,),
        ).fetchone()
        return {
            'pending': int(pending or 0),
            'scheduled': int(scheduled or 0),
            'action_required': int(action_required or 0),
            'latest': latest,
        }
    finally:
        conn.close()


def _format_request_type(value):
    labels = {
        'patient_appointment': 'Patient Appointment Request',
        'information_request': 'Additional Information Request',
        'consultation_request': 'Consultation Request',
        'audiometry_request': 'In-Person Audiometry Request',
        'specialist_referral': 'Specialist Referral',
        'acoustic_testing_invitation': 'Supervised Acoustic Testing',
    }
    return labels.get(value, value.replace('_', ' ').title() if value else 'Care Request')


def show_patient_care_page(patient_id):
    st.title('Appointments & Care Requests')

    if patient_id is None:
        st.error('This patient account is not linked to a patient record.')
        return

    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        summary = get_patient_care_summary(patient_id)
        c1, c2, c3 = st.columns(3)
        c1.metric('Open Requests', summary['pending'])
        c2.metric('Scheduled', summary['scheduled'])
        c3.metric('Action Required', summary['action_required'])

        if summary['action_required'] > 0:
            st.warning('Your audiologist has requested an action or response. Review the requests below.')
        elif summary['scheduled'] > 0:
            st.success('You have at least one scheduled care activity.')
        else:
            pass

        st.markdown('---')
        st.subheader('Request a consultation')
        with st.form('patient_new_consultation_form', clear_on_submit=True):
            mode = st.radio(
                'How would you like to speak with the audiologist?',
                ['In-person appointment', 'Video call', 'Phone call'],
                horizontal=True,
            )
            d = st.date_input('Preferred date', min_value=date.today())
            t = st.time_input('Preferred time', value=time(10, 0))
            reason = st.text_area(
                'Reason for the request',
                placeholder='Example: I would like more explanation about my tinnitus profile or next steps.',
            )
            submitted = st.form_submit_button('Send Request', use_container_width=True)
            if submitted:
                mode_map = {
                    'In-person appointment': 'in_person',
                    'Video call': 'video',
                    'Phone call': 'phone',
                }
                preferred = datetime.combine(d, t).strftime('%Y-%m-%d %H:%M')
                _insert_request(
                    conn,
                    patient_id,
                    'patient_to_audiologist',
                    'patient_appointment',
                    mode,
                    reason.strip() or 'Patient requested a consultation.',
                    communication_mode=mode_map[mode],
                    preferred_datetime=preferred,
                )
                st.success('Your request was sent to the audiologist.')
                st.rerun()

        st.markdown('---')
        st.subheader('Requests from your audiologist')
        incoming = conn.execute(
            """
            SELECT id, request_type, communication_mode, specialist, title, message,
                   preferred_datetime, scheduled_datetime, status, patient_response, audiologist_note, created_at
            FROM care_requests
            WHERE patient_id=? AND direction='audiologist_to_patient'
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()

        if not incoming:
            st.caption('No professional requests have been sent yet.')
        else:
            for row in incoming:
                (
                    request_id, request_type, communication_mode, specialist, title, message,
                    preferred_dt, scheduled_dt, status, patient_response, audiologist_note, created_at
                ) = row
                with st.container(border=True):
                    st.markdown(f"### {_format_request_type(request_type)}")
                    st.write(f"**Status:** {status}")
                    if title:
                        st.write(f"**Request:** {title}")
                    if message:
                        st.write(message)
                    if communication_mode:
                        st.caption(f"Mode: {communication_mode.replace('_', ' ').title()}")
                    if specialist:
                        st.caption(f"Referral specialist: {specialist}")
                    if scheduled_dt:
                        st.success(f"Scheduled: {scheduled_dt}")
                    elif preferred_dt:
                        st.caption(f"Proposed / preferred time: {preferred_dt}")
                    if audiologist_note:
                        st.caption(f"Professional note: {audiologist_note}")
                    if patient_response:
                        st.info(f"Your response: {patient_response}")

                    if request_type == 'information_request' and status not in ('Completed', 'Cancelled'):
                        response = st.text_area(
                            'Your response',
                            key=f'care_patient_response_{request_id}',
                            placeholder='Provide the additional information requested by your audiologist.',
                        )
                        if st.button('Send Information', key=f'care_send_info_{request_id}', use_container_width=True):
                            conn.execute(
                                """
                                UPDATE care_requests
                                SET patient_response=?, status='Responded', updated_at=?
                                WHERE id=?
                                """,
                                (response.strip(), _now(), request_id),
                            )
                            conn.commit()
                            st.success('Information sent.')
                            st.rerun()

                    if request_type in ('consultation_request', 'audiometry_request', 'acoustic_testing_invitation') and status not in ('Completed', 'Cancelled'):
                        a, b = st.columns(2)
                        if a.button('Confirm', key=f'care_confirm_{request_id}', use_container_width=True):
                            conn.execute(
                                "UPDATE care_requests SET status='Accepted', updated_at=? WHERE id=?",
                                (_now(), request_id),
                            )
                            conn.commit()
                            st.rerun()
                        if b.button('Request another time', key=f'care_reschedule_{request_id}', use_container_width=True):
                            conn.execute(
                                "UPDATE care_requests SET status='Needs Reschedule', updated_at=? WHERE id=?",
                                (_now(), request_id),
                            )
                            conn.commit()
                            st.rerun()

        st.markdown('---')
        st.subheader('Your requests')
        outgoing = conn.execute(
            """
            SELECT id, title, message, preferred_datetime, scheduled_datetime, status, created_at
            FROM care_requests
            WHERE patient_id=? AND direction='patient_to_audiologist'
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()
        if not outgoing:
            st.caption('You have not sent any consultation requests yet.')
        else:
            display = []
            for row in outgoing:
                display.append({
                    'Request': row[1],
                    'Preferred': row[3] or '—',
                    'Scheduled': row[4] or '—',
                    'Status': row[5],
                    'Created': row[6],
                })
            st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)
    finally:
        conn.close()


def show_audiologist_care_coordination(patient_id, patient_code=None):
    if patient_id is None:
        st.info('Select a patient first.')
        return

    conn = get_connection()
    try:
        ensure_care_coordination_table(conn)
        st.markdown('## Care Coordination')

        action = st.selectbox(
            'Professional next step',
            [
                'Request additional information',
                'Request phone / video / in-person consultation',
                'Request in-person audiometry',
                'Recommend specialist evaluation',
                'Invite for supervised acoustic testing',
            ],
            key=f'care_action_{patient_id}',
        )

        if action == 'Request additional information':
            details = st.text_area(
                'What information do you need?',
                key=f'care_info_details_{patient_id}',
                placeholder='Example: Please clarify when the tinnitus started and whether it changes with head movement.',
            )
            if st.button('Send Information Request', key=f'care_send_info_request_{patient_id}', use_container_width=True):
                _insert_request(
                    conn, patient_id, 'audiologist_to_patient', 'information_request',
                    'Additional information required', details.strip() or 'Please provide additional information before the next step.',
                    status='Needs Information',
                )
                st.success('Information request sent to the patient.')
                st.rerun()

        elif action == 'Request phone / video / in-person consultation':
            mode_label = st.radio('Consultation mode', ['Phone call', 'Video call', 'In-person'], horizontal=True, key=f'care_mode_{patient_id}')
            d = st.date_input('Proposed date', min_value=date.today(), key=f'care_consult_date_{patient_id}')
            t = st.time_input('Proposed time', value=time(10, 0), key=f'care_consult_time_{patient_id}')
            note = st.text_area('Reason / note', key=f'care_consult_note_{patient_id}')
            if st.button('Send Consultation Request', key=f'care_send_consult_{patient_id}', use_container_width=True):
                mode_map = {'Phone call': 'phone', 'Video call': 'video', 'In-person': 'in_person'}
                scheduled = datetime.combine(d, t).strftime('%Y-%m-%d %H:%M')
                _insert_request(
                    conn, patient_id, 'audiologist_to_patient', 'consultation_request',
                    mode_label, note.strip() or 'Professional consultation requested.',
                    communication_mode=mode_map[mode_label], scheduled_datetime=scheduled, status='Needs Confirmation',
                )
                st.success('Consultation request sent.')
                st.rerun()

        elif action == 'Request in-person audiometry':
            d = st.date_input('Proposed audiometry date', min_value=date.today(), key=f'care_audio_date_{patient_id}')
            t = st.time_input('Proposed time', value=time(10, 0), key=f'care_audio_time_{patient_id}')
            reason = st.text_area('Professional reason', key=f'care_audio_reason_{patient_id}', placeholder='Example: Calibrated hearing thresholds are required before supervised acoustic testing.')
            if st.button('Send Audiometry Request', key=f'care_send_audio_{patient_id}', use_container_width=True):
                scheduled = datetime.combine(d, t).strftime('%Y-%m-%d %H:%M')
                _insert_request(
                    conn, patient_id, 'audiologist_to_patient', 'audiometry_request',
                    'In-person audiometry requested', reason.strip() or 'Calibrated audiometry is requested before continuing.',
                    communication_mode='in_person', scheduled_datetime=scheduled, status='Needs Confirmation',
                )
                st.success('Audiometry request sent.')
                st.rerun()

        elif action == 'Recommend specialist evaluation':
            specialist = st.selectbox('Specialist', ['ENT / Otolaryngology', 'Neurology', 'General Practitioner', 'Other'], key=f'care_specialist_{patient_id}')
            reason = st.text_area(
                'Professional referral reason',
                key=f'care_referral_reason_{patient_id}',
                placeholder='Document why additional medical evaluation is recommended without making an automated diagnosis.',
            )
            if st.button('Create Referral', key=f'care_send_referral_{patient_id}', use_container_width=True):
                _insert_request(
                    conn, patient_id, 'audiologist_to_patient', 'specialist_referral',
                    'Additional specialist evaluation recommended',
                    reason.strip() or 'Further medical evaluation is recommended before continuing the SmartTinnitus acoustic workflow.',
                    specialist=specialist, status='Referred',
                )
                st.success('Referral information saved and shown to the patient.')
                st.rerun()

        else:
            d = st.date_input('Proposed testing date', min_value=date.today(), key=f'care_test_date_{patient_id}')
            t = st.time_input('Proposed time', value=time(10, 0), key=f'care_test_time_{patient_id}')
            note = st.text_area('Testing note', key=f'care_test_note_{patient_id}', placeholder='Example: Bring your usual headphones only if instructed; testing will be supervised by the audiologist.')
            if st.button('Invite Patient for Supervised Testing', key=f'care_send_test_{patient_id}', use_container_width=True):
                scheduled = datetime.combine(d, t).strftime('%Y-%m-%d %H:%M')
                _insert_request(
                    conn, patient_id, 'audiologist_to_patient', 'acoustic_testing_invitation',
                    'Supervised acoustic testing appointment', note.strip() or 'Please attend a supervised SmartTinnitus acoustic testing session.',
                    communication_mode='in_person', scheduled_datetime=scheduled, status='Needs Confirmation',
                )
                st.success('Testing invitation sent.')
                st.rerun()

        st.markdown('### Care history')
        rows = conn.execute(
            """
            SELECT id, direction, request_type, communication_mode, specialist, title, message,
                   preferred_datetime, scheduled_datetime, status, patient_response, audiologist_note, created_at
            FROM care_requests
            WHERE patient_id=?
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()

        if not rows:
            st.caption('No care-coordination records for this patient yet.')
        else:
            for row in rows:
                request_id = row[0]
                with st.expander(f"{_format_request_type(row[2])} — {row[9]} — {row[12]}"):
                    st.write(f"**Direction:** {'Patient → Audiologist' if row[1]=='patient_to_audiologist' else 'Audiologist → Patient'}")
                    if row[5]: st.write(f"**Title:** {row[5]}")
                    if row[6]: st.write(row[6])
                    if row[7]: st.caption(f"Preferred: {row[7]}")
                    if row[8]: st.caption(f"Scheduled: {row[8]}")
                    if row[10]: st.info(f"Patient response: {row[10]}")
                    status_options = ['Pending', 'Needs Information', 'Needs Confirmation', 'Responded', 'Accepted', 'Scheduled', 'Needs Reschedule', 'Completed', 'Cancelled', 'Referred']
                    current_status = row[9] if row[9] in status_options else 'Pending'
                    new_status = st.selectbox(
                        'Update status',
                        status_options,
                        index=status_options.index(current_status),
                        key=f'care_status_update_{request_id}'
                    )
                    scheduled_value = st.text_input(
                        'Scheduled / proposed date and time',
                        value=row[8] or '',
                        placeholder='YYYY-MM-DD HH:MM',
                        key=f'care_scheduled_update_{request_id}'
                    )
                    professional_note = st.text_area(
                        'Audiologist note',
                        value=row[11] or '',
                        key=f'care_aud_note_{request_id}'
                    )
                    if st.button('Save Care Update', key=f'care_save_update_{request_id}', use_container_width=True):
                        conn.execute(
                            """
                            UPDATE care_requests
                            SET status=?, scheduled_datetime=?, audiologist_note=?, updated_at=?
                            WHERE id=?
                            """,
                            (
                                new_status,
                                scheduled_value.strip() or None,
                                professional_note.strip() or None,
                                _now(),
                                request_id
                            ),
                        )
                        conn.commit()
                        st.success('Care record updated.')
                        st.rerun()
    finally:
        conn.close()
