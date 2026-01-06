"""
Utility functions for working with OpenAI Harmony format and GPT-OSS models.

This module provides display and parsing functions for Harmony-formatted
messages and responses, which use channels (analysis, commentary, final)
to separate reasoning, tool calls, and final outputs.
"""

from typing import Optional, List, Dict


def print_harmony_messages(messages):
    """
    Display Harmony messages in a human-readable format.
    
    Handles different content types:
    - SystemContent: model identity, reasoning effort, knowledge cutoff, channels
    - DeveloperContent: instructions
    - TextContent: user messages
    - ToolResponseContent: tool results
    
    Args:
        messages: List of Harmony Message objects
    
    Example:
        >>> messages = [Message.from_role_and_content(Role.SYSTEM, SystemContent.new())]
        >>> print_harmony_messages(messages)
    """
    for msg in messages:
        role = msg.author.role.value
        print(f"\n🔵 {role.upper()}")
        
        if isinstance(msg.content, list):
            for item in msg.content:
                # Handle SystemContent
                if hasattr(item, 'model_identity'):
                    print(f"   Identity: {item.model_identity}")
                if hasattr(item, 'reasoning_effort'):
                    print(f"   Reasoning: {item.reasoning_effort}")
                if hasattr(item, 'knowledge_cutoff'):
                    print(f"   Knowledge Cutoff: {item.knowledge_cutoff}")
                if hasattr(item, 'channel_config'):
                    print(f"   Valid Channels: {item.channel_config.valid_channels}")
                
                # Handle DeveloperContent
                if hasattr(item, 'instructions'):
                    print(f"   Instructions: {item.instructions}")
                
                # Handle TextContent (USER messages)
                if hasattr(item, 'text'):
                    print(f"   Text: {item.text}")
                
                # Handle ToolResponseContent
                if hasattr(item, 'tool_call_id'):
                    print(f"   Tool Call ID: {item.tool_call_id}")
                    if hasattr(item, 'content'):
                        print(f"   Result: {item.content}")
        else:
            print(f"   {msg.content}")
        print()


def display_harmony_response(response: str, show_tool_calls: bool = True):
    """
    Display a Harmony-formatted response broken down by channels.
    Shows the analysis, commentary, and final channels separately.
    
    Args:
        response: The raw response string from the model
        show_tool_calls: Whether to highlight potential tool calls
    
    Example:
        >>> response = generate(model, tokenizer, prompt, max_tokens=2048)
        >>> display_harmony_response(response)
    """
    print("\n" + "="*80)
    print("HARMONY RESPONSE BREAKDOWN")
    print("="*80)
    
    # Define the channels to look for
    channels = ['analysis', 'commentary', 'final']
    
    for channel in channels:
        marker = f"<|channel|>{channel}<|message|>"
        
        if marker in response:
            # Find all occurrences of this channel
            parts = response.split(marker)
            
            for idx, part in enumerate(parts[1:], 1):  # Skip first split (before first marker)
                # Extract content until next channel or end marker
                content = part.split("<|channel|>")[0]  # Stop at next channel
                content = content.split("<|end|>")[0]   # Stop at end marker
                content = content.strip()
                
                if content:
                    print(f"\n{'─'*80}")
                    
                    # Highlight tool calls in commentary/analysis channels
                    icon = "📍"
                    label = channel.upper()
                    
                    if show_tool_calls:
                        if channel == "commentary" and ("tool_calls" in content or "{" in content[:20]):
                            icon = "🔧"
                            label += " (LIKELY TOOL CALL)"
                        elif channel == "analysis" and "tool" in content.lower():
                            icon = "🔧"
                            label += " (POSSIBLE TOOL CALL)"
                    
                    print(f"{icon} CHANNEL: {label}")
                    
                    if len(parts) > 2:
                        print(f"   (Instance {idx})")
                    print(f"{'─'*80}")
                    print(content)
    
    print(f"\n{'='*80}\n")


def display_response_raw(response: str):
    """
    Display the raw response with channel markers highlighted for debugging.
    Useful for understanding the exact structure of the model output.
    
    Args:
        response: The raw response string from the model
    
    Example:
        >>> display_response_raw(response)
    """
    print("\n" + "="*80)
    print("RAW HARMONY RESPONSE (with highlighted markers)")
    print("="*80 + "\n")
    
    # Add visual markers to make channels obvious
    highlighted = response.replace("<|channel|>", "\n\n🔹 CHANNEL: ")
    highlighted = highlighted.replace("<|message|>", " 📝\n")
    highlighted = highlighted.replace("<|end|>", "\n━━━ END ━━━")
    highlighted = highlighted.replace("<|start|>", "\n\n▶️ START: ")
    
    print(highlighted)
    print("\n" + "="*80 + "\n")


def extract_channel_content(response: str, channel_name: str) -> Optional[str]:
    """
    Extract content from a specific channel in a Harmony response.
    
    Args:
        response: The raw response string from the model
        channel_name: The channel to extract ('analysis', 'commentary', or 'final')
    
    Returns:
        The content from the specified channel, or None if not found
    
    Example:
        >>> final_answer = extract_channel_content(response, 'final')
        >>> thinking = extract_channel_content(response, 'analysis')
    """
    marker = f"<|channel|>{channel_name}<|message|>"
    if marker in response:
        content = response.split(marker)[1].split("<|channel|>")[0].split("<|end|>")[0]
        return content.strip()
    return None


def extract_all_channels(response: str) -> Dict[str, List[str]]:
    """
    Extract content from all channels in a Harmony response.
    
    Args:
        response: The raw response string from the model
    
    Returns:
        Dictionary mapping channel names to lists of content strings
        (lists because a channel can appear multiple times)
    
    Example:
        >>> channels = extract_all_channels(response)
        >>> print(channels['final'][0])  # First final channel content
    """
    channels = {
        'analysis': [],
        'commentary': [],
        'final': []
    }
    
    for channel_name in channels.keys():
        marker = f"<|channel|>{channel_name}<|message|>"
        parts = response.split(marker)
        
        # Skip the first part (before any marker of this channel)
        for part in parts[1:]:
            content = part.split("<|channel|>")[0].split("<|end|>")[0].strip()
            if content:
                channels[channel_name].append(content)
    
    return channels


def get_final_response(response: str) -> str:
    """
    Extract just the final response channel content (user-facing answer).
    This is what you typically want to show to the end user.
    
    Args:
        response: The raw response string from the model
    
    Returns:
        The final channel content, or a message if not found
    
    Example:
        >>> answer = get_final_response(response)
        >>> print(answer)
    """
    final = extract_channel_content(response, 'final')
    if final:
        return final
    
    # If no final channel, check if model is still reasoning
    if extract_channel_content(response, 'analysis') or extract_channel_content(response, 'commentary'):
        return "[Model did not complete response - try increasing max_tokens]"
    
    return response


