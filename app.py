import streamlit as st
import sqlalchemy


# Initialize connection.
conn = st.connection("postgresql", type="sql")

# Perform query.
df = conn.query("""SELECT * FROM yahoo;""", ttl="10m")

# Print results.
for row in df.itertuples():
    st.write(f"{row.name} has a :{row.pet}:")