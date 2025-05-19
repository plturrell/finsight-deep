"""
Jena-based persistence layer for Digital Human Financial Advisor
Uses Apache Jena for RDF/OWL knowledge graph storage
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import uuid
from dataclasses import dataclass, asdict

from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore
import requests
from SPARQLWrapper import SPARQLWrapper, JSON, POST

from aiq.data_models.common import BaseModel
from aiq.utils.debugging_utils import log_function_call


# Define namespaces for the Digital Human ontology
DH = Namespace("http://aiqtoolkit.com/digital-human/")
FINANCE = Namespace("http://aiqtoolkit.com/finance/")
USER = Namespace("http://aiqtoolkit.com/user/")
SCHEMA = Namespace("http://schema.org/")

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """User profile data model"""
    user_id: str
    name: str
    email: str
    risk_tolerance: str
    investment_goals: List[str]
    created_at: datetime
    updated_at: datetime
    preferences: Dict[str, Any]


@dataclass
class Session:
    """Session data model"""
    session_id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    context: Dict[str, Any]
    interactions: List[Dict[str, Any]]
    emotional_states: List[Dict[str, Any]]


@dataclass
class Portfolio:
    """Portfolio data model"""
    portfolio_id: str
    user_id: str
    holdings: List[Dict[str, Any]]
    total_value: float
    performance: Dict[str, float]
    updated_at: datetime


class JenaPersistenceManager:
    """
    Manages data persistence using Apache Jena triple store
    Provides RDF/OWL knowledge graph storage for Digital Human data
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fuseki_url = config.get("fuseki_url", "http://localhost:3030")
        self.dataset_name = config.get("dataset_name", "digital_human")
        
        # Initialize RDF graph
        self.graph = Graph()
        self._init_namespaces()
        self._init_ontology()
        
        # Setup SPARQL endpoints
        self.query_endpoint = f"{self.fuseki_url}/{self.dataset_name}/query"
        self.update_endpoint = f"{self.fuseki_url}/{self.dataset_name}/update"
        
        # Initialize SPARQL wrappers
        self.sparql_query = SPARQLWrapper(self.query_endpoint)
        self.sparql_update = SPARQLWrapper(self.update_endpoint)
        self.sparql_update.setMethod(POST)
        
        logger.info(f"Initialized Jena persistence manager with Fuseki at {self.fuseki_url}")
    
    def _init_namespaces(self):
        """Initialize RDF namespaces"""
        self.graph.bind("dh", DH)
        self.graph.bind("finance", FINANCE)
        self.graph.bind("user", USER)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("xsd", XSD)
    
    def _init_ontology(self):
        """Initialize the Digital Human ontology"""
        # Define classes
        self.graph.add((DH.User, RDF.type, OWL.Class))
        self.graph.add((DH.Session, RDF.type, OWL.Class))
        self.graph.add((DH.Portfolio, RDF.type, OWL.Class))
        self.graph.add((DH.Interaction, RDF.type, OWL.Class))
        self.graph.add((DH.EmotionalState, RDF.type, OWL.Class))
        
        # Define properties
        self.graph.add((DH.hasProfile, RDF.type, OWL.ObjectProperty))
        self.graph.add((DH.hasSession, RDF.type, OWL.ObjectProperty))
        self.graph.add((DH.hasPortfolio, RDF.type, OWL.ObjectProperty))
        self.graph.add((DH.hasInteraction, RDF.type, OWL.ObjectProperty))
        self.graph.add((DH.hasEmotionalState, RDF.type, OWL.ObjectProperty))
        
        # Define data properties
        self.graph.add((DH.userId, RDF.type, OWL.DatatypeProperty))
        self.graph.add((DH.email, RDF.type, OWL.DatatypeProperty))
        self.graph.add((DH.riskTolerance, RDF.type, OWL.DatatypeProperty))
        self.graph.add((DH.createdAt, RDF.type, OWL.DatatypeProperty))
        self.graph.add((DH.updatedAt, RDF.type, OWL.DatatypeProperty))
        
        # Upload ontology to Fuseki
        self._upload_ontology()
    
    def _upload_ontology(self):
        """Upload the ontology to Fuseki"""
        try:
            ontology_data = self.graph.serialize(format="turtle")
            
            update_query = f"""
            PREFIX dh: <{DH}>
            
            INSERT DATA {{
                GRAPH <{DH}ontology> {{
                    {ontology_data}
                }}
            }}
            """
            
            self.sparql_update.setQuery(update_query)
            self.sparql_update.query()
            
            logger.info("Ontology uploaded to Fuseki successfully")
        except Exception as e:
            logger.error(f"Failed to upload ontology: {e}")
    
    async def create_user(self, user_profile: UserProfile) -> str:
        """Create a new user in the knowledge graph"""
        user_uri = USER[user_profile.user_id]
        
        update_query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        PREFIX schema: <{SCHEMA}>
        PREFIX xsd: <{XSD}>
        
        INSERT DATA {{
            <{user_uri}> a dh:User ;
                dh:userId "{user_profile.user_id}" ;
                schema:name "{user_profile.name}" ;
                schema:email "{user_profile.email}" ;
                dh:riskTolerance "{user_profile.risk_tolerance}" ;
                dh:createdAt "{user_profile.created_at.isoformat()}"^^xsd:dateTime ;
                dh:updatedAt "{user_profile.updated_at.isoformat()}"^^xsd:dateTime ;
                dh:preferences "{json.dumps(user_profile.preferences).replace('"', '\\"')}" .
        """
        
        # Add investment goals
        for goal in user_profile.investment_goals:
            update_query += f"""
                <{user_uri}> dh:hasInvestmentGoal "{goal}" .
            """
        
        update_query += "}"
        
        try:
            self.sparql_update.setQuery(update_query)
            self.sparql_update.query()
            
            logger.info(f"Created user: {user_profile.user_id}")
            return user_profile.user_id
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from the knowledge graph"""
        query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        PREFIX schema: <{SCHEMA}>
        
        SELECT ?name ?email ?riskTolerance ?createdAt ?updatedAt ?preferences
               (GROUP_CONCAT(?goal; SEPARATOR=",") AS ?goals)
        WHERE {{
            <{USER[user_id]}> a dh:User ;
                schema:name ?name ;
                schema:email ?email ;
                dh:riskTolerance ?riskTolerance ;
                dh:createdAt ?createdAt ;
                dh:updatedAt ?updatedAt ;
                dh:preferences ?preferences .
            OPTIONAL {{
                <{USER[user_id]}> dh:hasInvestmentGoal ?goal .
            }}
        }}
        GROUP BY ?name ?email ?riskTolerance ?createdAt ?updatedAt ?preferences
        """
        
        try:
            self.sparql_query.setQuery(query)
            self.sparql_query.setReturnFormat(JSON)
            results = self.sparql_query.query().convert()
            
            if results["results"]["bindings"]:
                result = results["results"]["bindings"][0]
                
                return UserProfile(
                    user_id=user_id,
                    name=result["name"]["value"],
                    email=result["email"]["value"],
                    risk_tolerance=result["riskTolerance"]["value"],
                    investment_goals=result["goals"]["value"].split(",") if result.get("goals") else [],
                    created_at=datetime.fromisoformat(result["createdAt"]["value"]),
                    updated_at=datetime.fromisoformat(result["updatedAt"]["value"]),
                    preferences=json.loads(result["preferences"]["value"])
                )
            
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def create_session(self, session: Session) -> str:
        """Create a new session in the knowledge graph"""
        session_uri = DH[f"session/{session.session_id}"]
        user_uri = USER[session.user_id]
        
        update_query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        PREFIX xsd: <{XSD}>
        
        INSERT DATA {{
            <{session_uri}> a dh:Session ;
                dh:sessionId "{session.session_id}" ;
                dh:userId "{session.user_id}" ;
                dh:startedAt "{session.started_at.isoformat()}"^^xsd:dateTime ;
                dh:context "{json.dumps(session.context).replace('"', '\\"')}" .
            
            <{user_uri}> dh:hasSession <{session_uri}> .
        """
        
        # Add interactions
        for i, interaction in enumerate(session.interactions):
            interaction_uri = DH[f"interaction/{session.session_id}/{i}"]
            update_query += f"""
                <{session_uri}> dh:hasInteraction <{interaction_uri}> .
                <{interaction_uri}> a dh:Interaction ;
                    dh:timestamp "{interaction['timestamp']}"^^xsd:dateTime ;
                    dh:type "{interaction['type']}" ;
                    dh:content "{json.dumps(interaction).replace('"', '\\"')}" .
            """
        
        # Add emotional states
        for i, state in enumerate(session.emotional_states):
            state_uri = DH[f"emotional_state/{session.session_id}/{i}"]
            update_query += f"""
                <{session_uri}> dh:hasEmotionalState <{state_uri}> .
                <{state_uri}> a dh:EmotionalState ;
                    dh:timestamp "{state['timestamp']}"^^xsd:dateTime ;
                    dh:emotion "{state['emotion']}" ;
                    dh:intensity {state['intensity']} .
            """
        
        update_query += "}"
        
        try:
            self.sparql_update.setQuery(update_query)
            self.sparql_update.query()
            
            logger.info(f"Created session: {session.session_id}")
            return session.session_id
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session data in the knowledge graph"""
        session_uri = DH[f"session/{session_id}"]
        timestamp = datetime.now(timezone.utc).isoformat()
        
        update_query = f"""
        PREFIX dh: <{DH}>
        PREFIX xsd: <{XSD}>
        
        DELETE {{
            <{session_uri}> dh:updatedAt ?oldUpdate .
        }}
        INSERT {{
            <{session_uri}> dh:updatedAt "{timestamp}"^^xsd:dateTime .
        """
        
        # Update session end time if provided
        if "ended_at" in updates:
            update_query += f"""
                <{session_uri}> dh:endedAt "{updates['ended_at'].isoformat()}"^^xsd:dateTime .
            """
        
        # Add new interactions
        if "interactions" in updates:
            for i, interaction in enumerate(updates["interactions"]):
                interaction_uri = DH[f"interaction/{session_id}/{uuid.uuid4()}"]
                update_query += f"""
                    <{session_uri}> dh:hasInteraction <{interaction_uri}> .
                    <{interaction_uri}> a dh:Interaction ;
                        dh:timestamp "{interaction['timestamp']}"^^xsd:dateTime ;
                        dh:type "{interaction['type']}" ;
                        dh:content "{json.dumps(interaction).replace('"', '\\"')}" .
                """
        
        update_query += """
        }
        WHERE {
            OPTIONAL { <""" + str(session_uri) + """> dh:updatedAt ?oldUpdate }
        }
        """
        
        try:
            self.sparql_update.setQuery(update_query)
            self.sparql_update.query()
            
            logger.info(f"Updated session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session from the knowledge graph"""
        query = f"""
        PREFIX dh: <{DH}>
        PREFIX xsd: <{XSD}>
        
        SELECT ?userId ?startedAt ?endedAt ?context 
               (GROUP_CONCAT(DISTINCT ?interaction; SEPARATOR="|") AS ?interactions)
               (GROUP_CONCAT(DISTINCT ?emotionalState; SEPARATOR="|") AS ?emotionalStates)
        WHERE {{
            <{DH[f"session/{session_id}"]}> a dh:Session ;
                dh:userId ?userId ;
                dh:startedAt ?startedAt ;
                dh:context ?context .
            OPTIONAL {{
                <{DH[f"session/{session_id}"]}> dh:endedAt ?endedAt .
            }}
            OPTIONAL {{
                <{DH[f"session/{session_id}"]}> dh:hasInteraction ?interactionUri .
                ?interactionUri dh:content ?interaction .
            }}
            OPTIONAL {{
                <{DH[f"session/{session_id}"]}> dh:hasEmotionalState ?stateUri .
                ?stateUri dh:emotion ?emotion ;
                         dh:intensity ?intensity ;
                         dh:timestamp ?stateTime .
                BIND(CONCAT("{{", 
                    '"emotion":"', ?emotion, '",',
                    '"intensity":', STR(?intensity), ',',
                    '"timestamp":"', STR(?stateTime), '"',
                    "}}") AS ?emotionalState)
            }}
        }}
        GROUP BY ?userId ?startedAt ?endedAt ?context
        """
        
        try:
            self.sparql_query.setQuery(query)
            self.sparql_query.setReturnFormat(JSON)
            results = self.sparql_query.query().convert()
            
            if results["results"]["bindings"]:
                result = results["results"]["bindings"][0]
                
                interactions = []
                if result.get("interactions"):
                    for interaction_json in result["interactions"]["value"].split("|"):
                        if interaction_json:
                            interactions.append(json.loads(interaction_json))
                
                emotional_states = []
                if result.get("emotionalStates"):
                    for state_json in result["emotionalStates"]["value"].split("|"):
                        if state_json:
                            emotional_states.append(json.loads(state_json))
                
                return Session(
                    session_id=session_id,
                    user_id=result["userId"]["value"],
                    started_at=datetime.fromisoformat(result["startedAt"]["value"]),
                    ended_at=datetime.fromisoformat(result["endedAt"]["value"]) if result.get("endedAt") else None,
                    context=json.loads(result["context"]["value"]),
                    interactions=interactions,
                    emotional_states=emotional_states
                )
            
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def create_portfolio(self, portfolio: Portfolio) -> str:
        """Create a new portfolio in the knowledge graph"""
        portfolio_uri = FINANCE[f"portfolio/{portfolio.portfolio_id}"]
        user_uri = USER[portfolio.user_id]
        
        update_query = f"""
        PREFIX dh: <{DH}>
        PREFIX finance: <{FINANCE}>
        PREFIX user: <{USER}>
        PREFIX xsd: <{XSD}>
        
        INSERT DATA {{
            <{portfolio_uri}> a finance:Portfolio ;
                finance:portfolioId "{portfolio.portfolio_id}" ;
                finance:userId "{portfolio.user_id}" ;
                finance:totalValue {portfolio.total_value} ;
                finance:updatedAt "{portfolio.updated_at.isoformat()}"^^xsd:dateTime ;
                finance:performance "{json.dumps(portfolio.performance).replace('"', '\\"')}" .
            
            <{user_uri}> dh:hasPortfolio <{portfolio_uri}> .
        """
        
        # Add holdings
        for i, holding in enumerate(portfolio.holdings):
            holding_uri = FINANCE[f"holding/{portfolio.portfolio_id}/{i}"]
            update_query += f"""
                <{portfolio_uri}> finance:hasHolding <{holding_uri}> .
                <{holding_uri}> a finance:Holding ;
                    finance:symbol "{holding['symbol']}" ;
                    finance:quantity {holding['quantity']} ;
                    finance:purchasePrice {holding['purchase_price']} ;
                    finance:currentPrice {holding['current_price']} ;
                    finance:purchaseDate "{holding['purchase_date']}"^^xsd:dateTime .
            """
        
        update_query += "}"
        
        try:
            self.sparql_update.setQuery(update_query)
            self.sparql_update.query()
            
            logger.info(f"Created portfolio: {portfolio.portfolio_id}")
            return portfolio.portfolio_id
        except Exception as e:
            logger.error(f"Failed to create portfolio: {e}")
            raise
    
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Session]:
        """Get recent sessions for a user"""
        query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        
        SELECT ?sessionId ?startedAt ?endedAt ?context
        WHERE {{
            <{USER[user_id]}> dh:hasSession ?session .
            ?session dh:sessionId ?sessionId ;
                    dh:startedAt ?startedAt ;
                    dh:context ?context .
            OPTIONAL {{
                ?session dh:endedAt ?endedAt .
            }}
        }}
        ORDER BY DESC(?startedAt)
        LIMIT {limit}
        """
        
        try:
            self.sparql_query.setQuery(query)
            self.sparql_query.setReturnFormat(JSON)
            results = self.sparql_query.query().convert()
            
            sessions = []
            for result in results["results"]["bindings"]:
                session = Session(
                    session_id=result["sessionId"]["value"],
                    user_id=user_id,
                    started_at=datetime.fromisoformat(result["startedAt"]["value"]),
                    ended_at=datetime.fromisoformat(result["endedAt"]["value"]) if result.get("endedAt") else None,
                    context=json.loads(result["context"]["value"]),
                    interactions=[],
                    emotional_states=[]
                )
                sessions.append(session)
            
            return sessions
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    async def execute_sparql_query(self, query: str) -> Dict[str, Any]:
        """Execute a custom SPARQL query"""
        try:
            self.sparql_query.setQuery(query)
            self.sparql_query.setReturnFormat(JSON)
            results = self.sparql_query.query().convert()
            return results
        except Exception as e:
            logger.error(f"Failed to execute SPARQL query: {e}")
            raise
    
    async def backup_database(self, backup_path: str):
        """Backup the entire knowledge graph"""
        query = """
        CONSTRUCT { ?s ?p ?o }
        WHERE { ?s ?p ?o }
        """
        
        try:
            self.sparql_query.setQuery(query)
            results = self.sparql_query.query().convert()
            
            # Serialize to file
            backup_graph = Graph()
            backup_graph.parse(data=results.serialize(format="turtle"), format="turtle")
            backup_graph.serialize(destination=backup_path, format="turtle")
            
            logger.info(f"Database backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise
    
    async def get_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics data for a user using SPARQL queries"""
        analytics = {}
        
        # Total sessions
        session_count_query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        
        SELECT (COUNT(?session) AS ?count)
        WHERE {{
            <{USER[user_id]}> dh:hasSession ?session .
        }}
        """
        
        # Average session duration
        avg_duration_query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        
        SELECT (AVG(?duration) AS ?avgDuration)
        WHERE {{
            <{USER[user_id]}> dh:hasSession ?session .
            ?session dh:startedAt ?start ;
                    dh:endedAt ?end .
            BIND((?end - ?start) AS ?duration)
        }}
        """
        
        # Most common emotions
        emotion_query = f"""
        PREFIX dh: <{DH}>
        PREFIX user: <{USER}>
        
        SELECT ?emotion (COUNT(?emotion) AS ?count)
        WHERE {{
            <{USER[user_id]}> dh:hasSession ?session .
            ?session dh:hasEmotionalState ?state .
            ?state dh:emotion ?emotion .
        }}
        GROUP BY ?emotion
        ORDER BY DESC(?count)
        LIMIT 5
        """
        
        try:
            # Execute queries
            session_count = await self.execute_sparql_query(session_count_query)
            analytics["total_sessions"] = int(session_count["results"]["bindings"][0]["count"]["value"])
            
            avg_duration = await self.execute_sparql_query(avg_duration_query)
            if avg_duration["results"]["bindings"]:
                analytics["avg_session_duration"] = float(avg_duration["results"]["bindings"][0]["avgDuration"]["value"])
            
            emotions = await self.execute_sparql_query(emotion_query)
            analytics["top_emotions"] = [
                {
                    "emotion": result["emotion"]["value"],
                    "count": int(result["count"]["value"])
                }
                for result in emotions["results"]["bindings"]
            ]
            
            return analytics
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}