// web/js/reload_image.js
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "reload_image_node",
    
    async setup() {
        console.log("Reload Image Node extension loaded with MutationObserver approach");
        
        // Create a MutationObserver to watch for new nodes being added to the DOM
        const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === 'childList') {
                    for (const node of mutation.addedNodes) {
                        if (node.classList && node.classList.contains('litegraph')) {
                            // Found a new node, check if it's our reload node
                            checkForReloadNodes();
                        }
                    }
                }
            }
        });
        
        // Start observing the entire document with the configured parameters
        observer.observe(document.body, { childList: true, subtree: true });
        
        // Also check for nodes on setup
        setTimeout(checkForReloadNodes, 1000);
        setTimeout(checkForReloadNodes, 3000);
        setTimeout(checkForReloadNodes, 5000);
        
        // Function to check for reload nodes and add buttons
        function checkForReloadNodes() {
            console.log("Checking for reload image nodes...");
            
            // Find nodes by their title
            const nodes = document.querySelectorAll('.litegraph .title');
            for (const titleElem of nodes) {
                if (titleElem.textContent.includes('Load Image (Reloadable)')) {
                    const nodeElem = titleElem.closest('.litegraph');
                    if (nodeElem && !nodeElem.querySelector('.reload-btn-added')) {
                        console.log("Found reload node:", nodeElem);
                        addReloadButton(nodeElem, titleElem);
                    }
                }
            }
        }
        
        // Function to add reload button to a node
        function addReloadButton(nodeElem, titleElem) {
            try {
                // Mark the node as processed
                nodeElem.classList.add('reload-btn-added');
                
                // Get node ID from the DOM
                const nodeId = nodeElem.id || '';
                console.log("Adding reload button to node ID:", nodeId);
                
                // Create button
                const reloadButton = document.createElement("button");
                reloadButton.innerText = "↻ RELOAD";
                reloadButton.className = "comfy-btn reload-image-btn";
                reloadButton.style.position = "absolute";
                reloadButton.style.top = "4px";
                reloadButton.style.right = "4px";
                reloadButton.style.zIndex = "9999";
                reloadButton.style.fontSize = "12px";
                reloadButton.style.padding = "4px 8px";
                reloadButton.style.backgroundColor = "#FF0000"; // Red for testing
                reloadButton.style.color = "white";
                reloadButton.style.border = "none";
                reloadButton.style.borderRadius = "3px";
                reloadButton.style.cursor = "pointer";
                reloadButton.style.fontWeight = "bold";
                
                // Handle click
                reloadButton.addEventListener("click", async (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    console.log("Reload button clicked for node:", nodeId);
                    
                    // Find the node object in the graph
                    const nodeObj = findNodeObjectById(nodeId);
                    if (nodeObj) {
                        // Execute the node
                        app.graph.runStep(1, [nodeObj]);
                        
                        // Visual feedback
                        reloadButton.style.backgroundColor = "#8B0000";
                        reloadButton.innerText = "⟳";
                        
                        setTimeout(() => {
                            reloadButton.innerText = "↻ RELOAD";
                            reloadButton.style.backgroundColor = "#FF0000";
                        }, 500);
                    } else {
                        console.error("Could not find node object for ID:", nodeId);
                        reloadButton.innerText = "ERROR";
                        
                        setTimeout(() => {
                            reloadButton.innerText = "↻ RELOAD";
                        }, 1500);
                    }
                });
                
                // Add to DOM - try both the title and the node element
                titleElem.style.position = "relative";
                titleElem.appendChild(reloadButton);
                
                console.log("Reload button added successfully to node:", nodeId);
            } catch (error) {
                console.error("Error adding reload button:", error);
            }
        }
        
        // Helper function to find a node object by DOM ID
        function findNodeObjectById(domId) {
            // Try to extract numeric ID from DOM ID
            const match = domId.match(/node_(\d+)/);
            if (match && match[1]) {
                const nodeId = parseInt(match[1]);
                return app.graph._nodes_by_id[nodeId] || app.graph._nodes.find(n => n.id === nodeId);
            }
            
            // Fallback - search all nodes
            return app.graph._nodes.find(node => {
                return node.domElement && node.domElement.id === domId;
            });
        }
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only modify our custom node
        if (nodeData.name !== "ReloadImageNode") {
            return;
        }
        
        console.log("Setting up ReloadImageNode definition with MutationObserver approach");
        
        // Add custom icon
        const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
            if (originalGetExtraMenuOptions) {
                originalGetExtraMenuOptions.apply(this, arguments);
            }
            
            // Add a custom icon - simple text for testing
            const iconContainer = this.domElement?.querySelector(".icon");
            if (iconContainer) {
                iconContainer.innerHTML = `
                    <div style="display:flex; align-items:center; justify-content:center; width:100%; height:100%; font-weight:bold; color:white;">
                        R
                    </div>
                `;
            }
        };
    }
});