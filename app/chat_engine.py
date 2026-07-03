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

    # def process_query_fallback(self, prompt: str, tenant_id: int, db: Session) -> dict:
    #     """Fallback when Groq client is unavailable. Returns a clear message instead of static hard‑coded answers."""
    #     return {
    #         "response": "Groq API not configured. Please set the GROQ_API_KEY environment variable.",
    #         "sql_executed": None,
    #         "raw_results": None
    #     }

    def _generate_sql_via_groq(self, prompt: str) -> dict:
        """Call Groq to generate a SQL query and explanation.
        Returns a dict with keys 'sql' and 'explanation'.
        """
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
        - You MUST filter every table by the placeholder `:tenant_id` to guarantee tenant isolation.
        - Respond ONLY with a JSON object: {"sql": "SELECT ...", "explanation": "..."}
        """
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": schema_description},
                {"role": "user", "content": f"User question: {prompt}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        print("response------>", response)
        raw_json = response.choices[0].message.content
        return json.loads(raw_json)

    def ask(self, prompt: str, tenant_id: int, db: Session) -> dict:
        """Process the natural language query. Uses Groq LLM if available, otherwise falls back to pattern matching."""
        if not self.client:
            # Groq client not configured – inform the user
            return {
                "response": "Groq API not configured. Please set the GROQ_API_KEY environment variable.",
                "sql_executed": None,
                "raw_results": None
            }

        try:
            # 1. Generate SQL via Groq
            groq_output = self._generate_sql_via_groq(prompt)
            sql_query = groq_output.get("sql")
            explanation = groq_output.get("explanation", "")

            # If Groq didn't produce SQL, return its explanation directly
            if not sql_query:
                return {
                    "response": explanation,
                    "sql_executed": None,
                    "raw_results": None
                }

            # 2. Validate safety and tenant isolation
            if not self.is_safe_select(sql_query):
                return {
                    "response": "The generated query was flagged as unsafe. Only read‑only SELECT operations are allowed.",
                    "sql_executed": sql_query,
                    "raw_results": None
                }
            if ":tenant_id" not in sql_query.lower():
                # Append simple filter if missing
                if "where" in sql_query.lower():
                    return {
                        "response": "Tenant isolation check failed. Please rephrase your query.",
                        "sql_executed": sql_query,
                        "raw_results": None
                    }
                else:
                    sql_query += " WHERE tenant_id = :tenant_id"

            # 3. Execute the safe SELECT
            db_result = db.execute(text(sql_query), {"tenant_id": tenant_id}).mappings().all()
            results = [dict(row) for row in db_result]

            # 4. Turn results into a friendly answer via Groq
            format_prompt = f"""
            The user asked: "{prompt}"
            The database was queried using SQL: "{sql_query}"
            The query returned these rows: {results}
            
            Write a concise, professional answer summarizing the results. If no rows were found, politely say so. Do not mention technical details unless asked.
            """
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful CRM assistant that summarizes query results into natural language."},
                    {"role": "user", "content": format_prompt}
                ],
                temperature=0.4
            )
            conversational_response = response.choices[0].message.content

            return {
                "response": conversational_response,
                "sql_executed": sql_query,
                "raw_results": results
            }
        except Exception as e:
            # Return a clear error instead of silently falling back
            return {
                "response": f"Groq API error: {e}. Please verify your GROQ_API_KEY and network connectivity.",
                "sql_executed": None,
                "raw_results": None
            }
