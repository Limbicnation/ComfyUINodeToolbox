import { app } from "../../scripts/app.js";

// Store node_id to node element mapping
const registeredNodes = new Map();

app.registerExtension({
    name: "reload_image_node",
    
    async setup() {
        // Listen for node registration events from the backend
        app.api.addEventListener("reload_image_node.register", (event) => {
            const { node_id, image_name } = event.detail;
            if (node_id && node_id !== "unknown") {
                registeredNodes.set(node_id, {
                    node_id: node_id,
                    image_name: image_name
                });
            }
        });
        
        // Listen for reload trigger events from the backend
        app.api.addEventListener("reload_image_node.reload", (event) => {
            const { node_id } = event.detail;
            if (node_id) {
                // Find the node in the workspace
                const node = app.graph.getNodeById(parseInt(node_id));
                if (node) {
                    // Trigger node execution
                    app.runNodeCommand(node);
                }
            }
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only modify our custom node type
        if (nodeData.name !== "ReloadImageNode") {
            return;
        }
        
        // Save the original onNodeCreated method
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        
        // Override the onNodeCreated method to add our custom UI
        nodeType.prototype.onNodeCreated = function() {
            // Call the original method first
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }
            
            // Pass the node_id to the backend
            const node_id = this.id.toString();
            this.properties.node_id = node_id;
            
            // Add hidden widget for node_id
            const widget = this.addWidget("text", "node_id", node_id, function(v) {}, { 
                hidden: true 
            });
            
            // Create reload button
            const reloadButton = document.createElement("button");
            reloadButton.innerText = "↻ Reload";
            reloadButton.className = "comfy-btn comfy-reload-btn";
            reloadButton.style.position = "absolute";
            reloadButton.style.top = "4px";
            reloadButton.style.right = "4px";
            reloadButton.style.zIndex = "1000";
            reloadButton.style.padding = "2px 5px";
            reloadButton.style.fontSize = "12px";
            
            // Add click event
            reloadButton.addEventListener("click", async (e) => {
                e.stopPropagation(); // Prevent node selection
                
                // Send reload request to the backend
                try {
                    const response = await fetch("/reload_image", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ node_id: node_id })
                    });
                    
                    if (!response.ok) {
                        console.error("Failed to reload image:", await response.text());
                    }
                } catch (error) {
                    console.error("Error reloading image:", error);
                }
            });
            
            // Add button to the node element
            this.domElement.appendChild(reloadButton);
        };
        
        // Add CSS styles for the reload button
        const style = document.createElement("style");
        style.textContent = `
            .comfy-reload-btn:hover {
                background-color: var(--comfy-input-bg);
                color: var(--input-text);
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }
});