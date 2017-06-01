import psycopg2

# code used to create event_predicts table in the eventdata database
conn = psycopg2.connect(dbname='eventdata', user='postgres', password='password', host='localhost')
c = conn.cursor()
c.execute(
    '''CREATE TABLE event_predicts (
            acct_type           CHAR(20),
            approx_payout_date  INT,
            body_length         INT,
            channels            INT,
            country             CHAR(2),
            currency            CHAR(3),
            delivery_method     FLOAT(8),
            description         CHAR(1000),
            email_domain        CHAR(20),
            event_created       INT,
            event_end           INT,
            event_published     FLOAT(8),
            event_start         INT,
            fb_published        INT,
            gts                 FLOAT(8),
            has_analytics       INT,
            has_header          FLOAT(8),
            has_logo            INT,
            listed              CHAR(1),
            name                CHAR(50),
            name_length         INT,
            num_order           INT,
            num_payouts         INT,
            object_id           INT,
            org_desc            CHAR(1000),
            org_name            CHAR(20),
            org_twitter         FLOAT(8),
            payee_name          CHAR(20),
            payout_type         CHAR(10),
            previous_payouts    CHAR(1000),
            sale_duration       FLOAT(8),
            sale_duration2      INT,
            show_map            INT,
            ticket_types        CHAR(1000),
            user_age            INT,
            user_created        INT,
            user_type           INT,
            venue_address       CHAR(100),
            venue_country       CHAR(5),
            venue_latitude      FLOAT(8),
            venue_longitude     FLOAT(8),
            venue_name          CHAR(50),
            venue_state         CHAR(5),
            fraud               FLOAT(8) )
    ;'''
)
conn.commit()
