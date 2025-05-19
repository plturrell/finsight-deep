#!/bin/bash

echo "🔄 Refreshing Elite Digital Human UI..."

# Force browser refresh
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - refresh Safari/Chrome
    osascript << EOF
    tell application "Google Chrome"
        set windowList to every window
        repeat with aWindow in windowList
            set tabList to every tab of aWindow
            repeat with atab in tabList
                if (url of atab contains "localhost:8080") then
                    tell atab to reload
                end if
            end repeat
        end repeat
    end tell
    
    tell application "Safari"
        try
            set windowList to every window
            repeat with aWindow in windowList
                set tabList to every tab of aWindow
                repeat with atab in tabList
                    if (url of atab contains "localhost:8080") then
                        tell atab to do JavaScript "window.location.reload()"
                    end if
                end repeat
            end repeat
        end try
    end tell
EOF
    echo "✅ Browser refreshed!"
else
    echo "Please manually refresh your browser (Ctrl+R or Cmd+R)"
fi

echo ""
echo "🎯 What to test:"
echo "1. The avatar should now show a human-like head with facial features"
echo "2. Type a message in the chat - it should work now"
echo "3. The avatar's eyes should follow your mouse cursor"
echo "4. Click the quick action buttons to test interactions"