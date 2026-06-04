import os
import re
import json
from sqlalchemy import text
from sqlalchemy.orm import Session
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"

class CRMQueryEngine:
    def __init__(self):
        self.client = None
        if GROQ_API_KEY and GROQ_API_KEY.strip():
            try:
                self.client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                self.client = None

    def is_safe_select(self, query: str) -> bool:
        """Verify that the query is a SELECT query and contains no data modifying commands."""
        clean_q = query.strip().lower()
        if not clean_q.startswith("select"):
            return False
        
        # Forbidden keywords that modify data or alter database structure
        forbidden = [
            "insert", "update", "delete", "drop", "truncate", "alter", 
            "create", "grant", "revoke", "replace", "vacuum", "analyze", 
            "explain", "begin", "commit", "rollback", "into"
        ]
        for word in forbidden:
            if re.search(r'\b' + word + r'\b', clean_q):
                return False
        return True

    def process_query_fallback(self, prompt: str, tenant_id: int, db: Session) -> dict:
        """Fallback rule-based parser that executes predefined SQL queries for key patterns."""
        p = prompt.lower().strip()
        
        # 1. Lead Count Queries
        if any(x in p for x in ["how many leads", "leads count", "number of leads", "leads available"]):
            sql = "SELECT COUNT(*) as count FROM leads WHERE tenant_id = :tenant_id"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().first()
            count = res["count"] if res else 0
            return {
                "response": f"You currently have {count} leads in your database.",
                "sql_executed": sql,
                "raw_results": [{"count": count}]
            }
            
        # 2. Lead Value Queries
        elif any(x in p for x in ["leads value", "total lead value", "worth of leads", "value of leads"]):
            sql = "SELECT COALESCE(SUM(value), 0) as total_value FROM leads WHERE tenant_id = :tenant_id"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().first()
            val = float(res["total_value"]) if res else 0.0
            return {
                "response": f"The total value of your active leads is ${val:,.2f}.",
                "sql_executed": sql,
                "raw_results": [{"total_value": val}]
            }

        # 3. Client Count Queries
        elif any(x in p for x in ["how many clients", "clients count", "number of clients", "total clients"]):
            sql = "SELECT COUNT(*) as count FROM clients WHERE tenant_id = :tenant_id"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().first()
            count = res["count"] if res else 0
            return {
                "response": f"You have a total of {count} registered clients.",
                "sql_executed": sql,
                "raw_results": [{"count": count}]
            }

        # 4. Opportunity Count Queries
        elif any(x in p for x in ["how many opportunities", "opportunities count", "number of opportunities"]):
            sql = "SELECT COUNT(*) as count FROM opportunities WHERE tenant_id = :tenant_id"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().first()
            count = res["count"] if res else 0
            return {
                "response": f"There are {count} opportunities in your sales pipeline.",
                "sql_executed": sql,
                "raw_results": [{"count": count}]
            }

        # 5. Opportunity Value Queries
        elif any(x in p for x in ["opportunities value", "value of opportunities", "total opportunity value", "pipeline value", "revenue pipeline"]):
            sql = "SELECT COALESCE(SUM(value), 0) as total_value FROM opportunities WHERE tenant_id = :tenant_id"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().first()
            val = float(res["total_value"]) if res else 0.0
            return {
                "response": f"The total value of your sales pipeline is ${val:,.2f}.",
                "sql_executed": sql,
                "raw_results": [{"pipeline_value": val}]
            }

        # 6. List Leads Queries
        elif any(x in p for x in ["list leads", "show leads", "show me leads", "recent leads"]):
            sql = "SELECT title, status, value FROM leads WHERE tenant_id = :tenant_id ORDER BY created_at DESC LIMIT 5"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().all()
            results = [dict(r) for r in res]
            if not results:
                return {"response": "No leads found for your tenant.", "sql_executed": sql, "raw_results": []}
            
            lead_list = "\n".join([f"- **{r['title']}** ({r['status']}) - ${float(r['value']):,.2f}" for r in results])
            return {
                "response": f"Here are your most recent leads:\n\n{lead_list}",
                "sql_executed": sql,
                "raw_results": results
            }

        # 7. List Clients Queries
        elif any(x in p for x in ["list clients", "show clients", "show me clients", "registered clients"]):
            sql = "SELECT name, company, email FROM clients WHERE tenant_id = :tenant_id ORDER BY name ASC LIMIT 5"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().all()
            results = [dict(r) for r in res]
            if not results:
                return {"response": "No clients found for your tenant.", "sql_executed": sql, "raw_results": []}
            
            client_list = "\n".join([f"- **{r['name']}** at *{r['company'] or 'N/A'}* ({r['email'] or 'No email'})" for r in results])
            return {
                "response": f"Here are your clients:\n\n{client_list}",
                "sql_executed": sql,
                "raw_results": results
            }

        # 8. List Opportunities Queries
        elif any(x in p for x in ["list opportunities", "show opportunities", "show me opportunities", "recent opportunities"]):
            sql = "SELECT title, stage, value FROM opportunities WHERE tenant_id = :tenant_id ORDER BY created_at DESC LIMIT 5"
            res = db.execute(text(sql), {"tenant_id": tenant_id}).mappings().all()
            results = [dict(r) for r in res]
            if not results:
                return {"response": "No opportunities found for your tenant.", "sql_executed": sql, "raw_results": []}
            
            opp_list = "\n".join([f"- **{r['title']}** ({r['stage']}) - ${float(r['value']):,.2f}" for r in results])
            return {
                "response": f"Here are your latest opportunities:\n\n{opp_list}",
                "sql_executed": sql,
                "raw_results": results
            }

        # Default Response
        return {
            "response": "Hello! I can help you search your CRM data. Try asking questions like:\n"
                        "- *'How many leads do we have?'*\n"
                        "- *'What is the total value of our opportunities?'*\n"
                        "- *'Show me our recent clients'*\n"
                        "- *'List leads'*",
            "sql_executed": None,
            "raw_results": None
        }

    def ask(self, prompt: str, tenant_id: int, db: Session) -> dict:
        """Processes the natural language query. Uses Groq LLM if available, otherwise fallbacks to pattern matching."""
        if not self.client:
            return self.process_query_fallback(prompt, tenant_id, db)

        # 1. Text-to-SQL Step using Groq
        schema_description = """
        Database Schema (PostgreSQL):
        
        1. Table: clients
           Columns: id (INTEGER, PK), tenant_id (INTEGER, FK), name (VARCHAR), email (VARCHAR), phone (VARCHAR), company (VARCHAR), created_at (TIMESTAMP)
        
        2. Table: leads
           Columns: id (INTEGER, PK), tenant_id (INTEGER, FK), client_id (INTEGER, FK to clients), name_fallback (VARCHAR), email_fallback (VARCHAR), title (VARCHAR), description (TEXT), status (VARCHAR: 'New', 'Contacted', 'Qualified', 'Lost', 'Won'), value (NUMERIC), created_at (TIMESTAMP)
        
        3. Table: opportunities
           Columns: id (INTEGER, PK), tenant_id (INTEGER, FK), lead_id (INTEGER, FK to leads), title (VARCHAR), description (TEXT), stage (VARCHAR: 'Discovery', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost'), value (NUMERIC), probability (INTEGER 0-100), close_date (DATE), created_at (TIMESTAMP)
        
        4. Table: users
           Columns: id (INTEGER, PK), tenant_id (INTEGER, FK), email (VARCHAR), name (VARCHAR), role (VARCHAR: 'admin', 'user'), created_at (TIMESTAMP)
        
        CRITICAL RULES:
        - Only generate SELECT queries. Never generate INSERT, UPDATE, DELETE, or ALTER.
        - You MUST filter every table by the placeholder `:tenant_id` to guarantee tenant isolation. For example, `leads.tenant_id = :tenant_id`.
        - If joining tables, ensure both have their tenant_id filtered or the join joins on their keys and is isolated.
        - Format query parameters using ':tenant_id' exactly.
        - Respond ONLY with a JSON object of format:
          {
             "sql": "SELECT ... WHERE tenant_id = :tenant_id",
             "explanation": "Short description of query"
          }
        - If the user's question cannot be answered by querying the database, set "sql" to null and explain why in "explanation".
        """

        try:
            # Call Groq to generate SQL
            sql_response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": schema_description},
                    {"role": "user", "content": f"User question: {prompt}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            raw_json = sql_response.choices[0].message.content
            parsed = json.loads(raw_json)
            sql_query = parsed.get("sql")
            explanation = parsed.get("explanation", "")

            # If no SQL is generated, return the explanation directly
            if not sql_query:
                return {
                    "response": explanation,
                    "sql_executed": None,
                    "raw_results": None
                }

            # Security and Tenant Isolation validation
            if not self.is_safe_select(sql_query):
                return {
                    "response": "The query generated was flagged as unsafe. I can only perform read-only SELECT operations.",
                    "sql_executed": sql_query,
                    "raw_results": None
                }

            # Enforce that the word ':tenant_id' must be in the SQL to prevent tenant leaks
            if ":tenant_id" not in sql_query.lower():
                # Try to append if it's simple, or reject
                if "where" in sql_query.lower():
                    # We can try to append, but safer to reject to protect data
                    return {
                        "response": "Tenant isolation check failed. Please rephrase your query.",
                        "sql_executed": sql_query,
                        "raw_results": None
                    }
                else:
                    # Append WHERE clause
                    sql_query += " WHERE tenant_id = :tenant_id"

            # Execute the SQL safely
            db_result = db.execute(text(sql_query), {"tenant_id": tenant_id}).mappings().all()
            results = [dict(row) for row in db_result]

            # 2. SQL-to-Text: Format the response using the LLM
            format_prompt = f"""
            The user asked: "{prompt}"
            The database was queried using SQL: "{sql_query}"
            The query returned the following rows: {results}
            
            Write a clear, concise, and professional answer summarizing these results. 
            If the results are empty, politely state that no matching records were found.
            Do not mention technical details like SQL, keys, tables, or database fields unless asked.
            Format numbers, values (as currency), and dates beautifully.
            """

            response_format = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful CRM sales assistant. Summarize database results into natural language responses."},
                    {"role": "user", "content": format_prompt}
                ],
                temperature=0.4
            )

            conversational_response = response_format.choices[0].message.content

            return {
                "response": conversational_response,
                "sql_executed": sql_query,
                "raw_results": results
            }

        except Exception as e:
            print(f"Groq API error: {e}. Falling back to rule-based engine.")
            return self.process_query_fallback(prompt, tenant_id, db)
