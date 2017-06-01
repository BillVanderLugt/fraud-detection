from build_model import Model, get_data
import pickle
import pandas as pd
import psycopg2

def predict_and_store(record,model,conn):
    '''
    Function that takes in user data and stores it to a database with a prediction as to whether or not the event listing is fraudulent. Returns said prediction.

    params:
    record - The user's inputted data, in simple, tuple form
    model - The unpickled model, already trained and ready to predict
    conn - A psycopg2 psql connection to database 'eventdata'

    output:
    prob_of_fraud - The probability that the event listing the user supplied data for is fraudulent.

    '''
    c = conn.cursor()

    # check if table event_predicts has any records (ie; if it has been created)
    # if not, create the table and its schema
    c.execute('SELECT * FROM information_schema.tables WHERE table_name=%s',('event_predicts'))
    if not c.fetchone()[0]:
        c.execute(
            '''CREATE TABLE event_predicts (
            acct_type           CHAR(20),
            approx_payout_date  INT,
            body_length         INT,
            channels            INT,
            country             CHAR(2),
            currency            CHAR(3),
            delivery_method     FLOAT(8),
            description         CHAR(50000),
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

    # predict on record
    columns = ['body_length','sale_duration2','user_age','name_length','payee_ind','user_type','fb_published']
    X = pd.DataFrame.from_records([record],columns=columns)
    prob_of_fraud = model.predict(X)[0][1]
    with_prediction = tuple(list(record).append(prob_of_fraud))


    columns = ['body_length','sale_duration2','user_age','name_length','payee_name','user_type','fb_published']
    # insert record + predicted fraud probability into event_predicts
    c.execute('INSERT INTO event_predicts {} VALUES {}'.format(tuple(columns.append('fraud')),with_prediction))

    # commit changes to the database
    conn.commit()

    return prob_of_fraud
