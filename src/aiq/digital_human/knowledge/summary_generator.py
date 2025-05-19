"""
Summary Generator for Financial Documents
Creates concise, accurate summaries using DSPy
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

import dspy
from dspy.predict import Predict
from dspy.modules import ChainOfThought
from dspy.teleprompt import BootstrapFewShot
from transformers import pipeline

from aiq.data_models.common import BaseModel
from aiq.utils.debugging_utils import log_function_call


logger = logging.getLogger(__name__)


# DSPy signatures for summarization
class FinancialSummarization(dspy.Signature):
    """Generate concise financial summaries."""
    document_text = dspy.InputField(desc="Financial document text")
    document_type = dspy.InputField(desc="Type of financial document")
    target_length = dspy.InputField(desc="Target summary length in words")
    focus_areas = dspy.InputField(desc="Key areas to focus on")
    
    summary = dspy.OutputField(desc="Concise financial summary")
    key_points = dspy.OutputField(desc="Bullet points of key information")
    metrics_highlighted = dspy.OutputField(desc="Important metrics and values")


class ExecutiveSummary(dspy.Signature):
    """Generate executive-level financial summaries."""
    detailed_content = dspy.InputField(desc="Detailed financial information")
    audience_level = dspy.InputField(desc="Target audience expertise level")
    
    executive_summary = dspy.OutputField(desc="High-level executive summary")
    strategic_insights = dspy.OutputField(desc="Strategic implications")
    action_items = dspy.OutputField(desc="Recommended actions")
    risk_assessment = dspy.OutputField(desc="Key risks identified")


class ComparativeSummary(dspy.Signature):
    """Generate comparative financial summaries."""
    documents = dspy.InputField(desc="List of financial documents to compare")
    comparison_criteria = dspy.InputField(desc="Criteria for comparison")
    
    comparative_summary = dspy.OutputField(desc="Comparative analysis summary")
    similarities = dspy.OutputField(desc="Key similarities identified")
    differences = dspy.OutputField(desc="Key differences identified")
    trends = dspy.OutputField(desc="Trends across documents")


@dataclass
class Summary:
    """Generated summary with metadata"""
    summary_id: str
    summary_type: str
    content: str
    key_points: List[str]
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    confidence_score: float


class SummaryGenerator:
    """
    Advanced summary generator for financial documents
    Uses DSPy for prompt optimization and multi-level summarization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize DSPy
        self.lm = dspy.OpenAI(
            model=config.get("model", "gpt-4"),
            api_key=config.get("api_key"),
            temperature=config.get("temperature", 0.3),
            max_tokens=config.get("max_tokens", 1000)
        )
        dspy.settings.configure(lm=self.lm)
        
        # Initialize summarization modules
        self.basic_summarizer = ChainOfThought(FinancialSummarization)
        self.executive_summarizer = Predict(ExecutiveSummary)
        self.comparative_summarizer = Predict(ComparativeSummary)
        
        # Initialize transformer models for additional processing
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=config.get("sentiment_model", "finbert-tone")
        )
        
        # Summary templates
        self.summary_templates = {
            'earnings': {
                'focus_areas': ['revenue', 'eps', 'guidance', 'key metrics'],
                'target_length': 150
            },
            'research': {
                'focus_areas': ['recommendation', 'target price', 'risks', 'catalysts'],
                'target_length': 200
            },
            'financial_statement': {
                'focus_areas': ['key ratios', 'year-over-year changes', 'trends'],
                'target_length': 175
            }
        }
        
        # Summary cache
        self.summary_cache = {}
        self.cache_duration = config.get("cache_duration", 3600)
        
        # Metrics
        self.metrics = {
            'summaries_generated': 0,
            'executive_summaries': 0,
            'comparative_summaries': 0,
            'average_confidence': 0.0
        }
        
        logger.info("Initialized Summary Generator")
    
    async def generate_summary(
        self,
        text: str,
        document_type: str = 'general',
        target_length: Optional[int] = None,
        summary_type: str = 'basic'
    ) -> Summary:
        """
        Generate summary of financial text
        
        Args:
            text: Text to summarize
            document_type: Type of financial document
            target_length: Target length in words
            summary_type: Type of summary (basic, executive, detailed)
            
        Returns:
            Generated summary
        """
        # Check cache
        cache_key = f"{hash(text)}_{document_type}_{summary_type}"
        if cache_key in self.summary_cache:
            cached = self.summary_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < self.cache_duration:
                logger.debug("Returning cached summary")
                return cached['summary']
        
        try:
            # Select appropriate summarizer
            if summary_type == 'executive':
                summary = await self._generate_executive_summary(text, document_type)
            elif summary_type == 'comparative':
                summary = await self._generate_comparative_summary([text], document_type)
            else:
                summary = await self._generate_basic_summary(text, document_type, target_length)
            
            # Cache result
            self.summary_cache[cache_key] = {
                'summary': summary,
                'timestamp': datetime.now()
            }
            
            # Update metrics
            self.metrics['summaries_generated'] += 1
            self.metrics['average_confidence'] = (
                self.metrics['average_confidence'] * 0.9 + 
                summary.confidence_score * 0.1
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            raise
    
    async def _generate_basic_summary(
        self,
        text: str,
        document_type: str,
        target_length: Optional[int]
    ) -> Summary:
        """Generate basic financial summary"""
        # Get template
        template = self.summary_templates.get(
            document_type,
            {'focus_areas': ['key information'], 'target_length': 150}
        )
        
        # Prepare input
        summary_input = {
            'document_text': text,
            'document_type': document_type,
            'target_length': str(target_length or template['target_length']),
            'focus_areas': ', '.join(template['focus_areas'])
        }
        
        # Generate summary
        result = self.basic_summarizer(**summary_input)
        
        # Parse results
        summary_content = result.summary
        key_points = self._parse_key_points(result.key_points)
        metrics = self._parse_metrics(result.metrics_highlighted)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(summary_content)
        
        # Calculate confidence
        confidence = self._calculate_confidence(summary_content, key_points, metrics)
        
        # Create summary object
        summary = Summary(
            summary_id=f"sum_{datetime.now().timestamp()}",
            summary_type='basic',
            content=summary_content,
            key_points=key_points,
            metrics=metrics,
            metadata={
                'document_type': document_type,
                'target_length': target_length,
                'actual_length': len(summary_content.split()),
                'sentiment': sentiment
            },
            created_at=datetime.now(),
            confidence_score=confidence
        )
        
        return summary
    
    async def _generate_executive_summary(
        self,
        text: str,
        document_type: str
    ) -> Summary:
        """Generate executive-level summary"""
        # First generate detailed summary
        basic_summary = await self._generate_basic_summary(text, document_type, 200)
        
        # Prepare executive summary input
        exec_input = {
            'detailed_content': json.dumps({
                'summary': basic_summary.content,
                'key_points': basic_summary.key_points,
                'metrics': basic_summary.metrics
            }),
            'audience_level': 'executive'
        }
        
        # Generate executive summary
        result = self.executive_summarizer(**exec_input)
        
        # Parse results
        exec_content = result.executive_summary
        strategic_insights = self._parse_list(result.strategic_insights)
        action_items = self._parse_list(result.action_items)
        risks = self._parse_list(result.risk_assessment)
        
        # Create summary object
        summary = Summary(
            summary_id=f"exec_{datetime.now().timestamp()}",
            summary_type='executive',
            content=exec_content,
            key_points=strategic_insights,
            metrics=basic_summary.metrics,
            metadata={
                'document_type': document_type,
                'action_items': action_items,
                'risks': risks,
                'base_summary_id': basic_summary.summary_id
            },
            created_at=datetime.now(),
            confidence_score=basic_summary.confidence_score * 0.95
        )
        
        self.metrics['executive_summaries'] += 1
        
        return summary
    
    async def _generate_comparative_summary(
        self,
        texts: List[str],
        document_type: str
    ) -> Summary:
        """Generate comparative summary across multiple documents"""
        # Prepare input
        comp_input = {
            'documents': json.dumps(texts),
            'comparison_criteria': f"{document_type} comparison"
        }
        
        # Generate comparative summary
        result = self.comparative_summarizer(**comp_input)
        
        # Parse results
        comp_content = result.comparative_summary
        similarities = self._parse_list(result.similarities)
        differences = self._parse_list(result.differences)
        trends = self._parse_list(result.trends)
        
        # Create summary object
        summary = Summary(
            summary_id=f"comp_{datetime.now().timestamp()}",
            summary_type='comparative',
            content=comp_content,
            key_points=trends,
            metrics={
                'documents_compared': len(texts),
                'similarities_found': len(similarities),
                'differences_found': len(differences)
            },
            metadata={
                'document_type': document_type,
                'similarities': similarities,
                'differences': differences
            },
            created_at=datetime.now(),
            confidence_score=0.85
        )
        
        self.metrics['comparative_summaries'] += 1
        
        return summary
    
    def _parse_key_points(self, key_points_str: str) -> List[str]:
        """Parse key points from string"""
        try:
            if key_points_str.startswith('['):
                return json.loads(key_points_str)
            
            # Parse bullet points
            points = []
            for line in key_points_str.split('\n'):
                line = line.strip()
                if line and line[0] in ['•', '-', '*', '·']:
                    points.append(line[1:].strip())
                elif line:
                    points.append(line)
            
            return points
        except Exception:
            return [key_points_str]
    
    def _parse_metrics(self, metrics_str: str) -> Dict[str, Any]:
        """Parse metrics from string"""
        try:
            if metrics_str.startswith('{'):
                return json.loads(metrics_str)
            
            metrics = {}
            
            # Parse key-value pairs
            import re
            pattern = r'([\w\s]+):\s*\$?([\d,]+(?:\.\d+)?%?)'
            
            for match in re.finditer(pattern, metrics_str):
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Convert to appropriate type
                if value.endswith('%'):
                    metrics[key] = float(value.rstrip('%'))
                elif ',' in value:
                    metrics[key] = float(value.replace(',', ''))
                else:
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
            
            return metrics
        except Exception:
            return {}
    
    def _parse_list(self, list_str: str) -> List[str]:
        """Parse list from string"""
        try:
            if list_str.startswith('['):
                return json.loads(list_str)
            
            # Parse lines
            items = []
            for line in list_str.split('\n'):
                line = line.strip()
                if line:
                    # Remove bullets/numbers
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^[•\-*·]\s*', '', line)
                    items.append(line)
            
            return items
        except Exception:
            return [list_str]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text"""
        try:
            results = self.sentiment_analyzer(text[:512])  # Limit length
            
            sentiment_scores = {}
            for result in results:
                sentiment_scores[result['label'].lower()] = result['score']
            
            return sentiment_scores
        except Exception:
            return {'neutral': 1.0}
    
    def _calculate_confidence(
        self,
        content: str,
        key_points: List[str],
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for summary"""
        confidence = 0.5  # Base confidence
        
        # Content quality
        if len(content) > 50:
            confidence += 0.1
        
        if key_points and len(key_points) > 2:
            confidence += 0.2
        
        if metrics and len(metrics) > 1:
            confidence += 0.2
        
        # Check for completeness
        if any(keyword in content.lower() for keyword in ['revenue', 'profit', 'growth']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def generate_multi_level_summary(
        self,
        text: str,
        document_type: str = 'general'
    ) -> Dict[str, Summary]:
        """Generate summaries at multiple levels of detail"""
        summaries = {}
        
        # Generate basic summary
        summaries['basic'] = await self.generate_summary(
            text,
            document_type,
            target_length=100,
            summary_type='basic'
        )
        
        # Generate detailed summary
        summaries['detailed'] = await self.generate_summary(
            text,
            document_type,
            target_length=300,
            summary_type='basic'
        )
        
        # Generate executive summary
        summaries['executive'] = await self.generate_summary(
            text,
            document_type,
            summary_type='executive'
        )
        
        return summaries
    
    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 200
    ) -> Summary:
        """Summarize a conversation thread"""
        # Combine messages
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        # Generate summary with special handling for conversations
        summary_input = {
            'document_text': conversation_text,
            'document_type': 'conversation',
            'target_length': str(max_length),
            'focus_areas': 'key decisions, action items, questions answered'
        }
        
        result = self.basic_summarizer(**summary_input)
        
        # Create summary object
        summary = Summary(
            summary_id=f"conv_{datetime.now().timestamp()}",
            summary_type='conversation',
            content=result.summary,
            key_points=self._parse_key_points(result.key_points),
            metrics={'message_count': len(messages)},
            metadata={
                'conversation_length': len(messages),
                'participants': list(set(msg['role'] for msg in messages))
            },
            created_at=datetime.now(),
            confidence_score=0.9
        )
        
        return summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generator metrics"""
        return {
            'summaries_generated': self.metrics['summaries_generated'],
            'executive_summaries': self.metrics['executive_summaries'],
            'comparative_summaries': self.metrics['comparative_summaries'],
            'average_confidence': self.metrics['average_confidence'],
            'cache_size': len(self.summary_cache)
        }