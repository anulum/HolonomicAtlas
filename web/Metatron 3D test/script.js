document.addEventListener('DOMContentLoaded', () => {

    // --- DATA ARCHITECTURE ---

    const scpnData = {
        '1': {
            title: "Quantum Biological Substrate",
            purpose: "To establish the foundational layer where quantum phenomena in biological systems give rise to proto-conscious events.",
            components: [
                "Microtubule Q-bits",
                "Orchestrated Objective Reduction (Orch-OR)",
                "Quantum Tunneling in Enzymes"
            ],
            connections: "Connects directly to Layer 2, providing the raw data for neural processing. Its stability is influenced by cosmological constants from Layer 13.",
            connectedLayers: [2, 13]
        },
        '2': {
            title: "Neural Network Processing",
            purpose: "To process the proto-conscious events from the quantum layer into coherent neural signals and patterns.",
            components: [
                "Synaptic Plasticity",
                "Action Potentials",
                "Neural Oscillation Patterns"
            ],
            connections: "Receives input from Layer 1 and sends processed information to Layer 3 for subjective experience integration.",
            connectedLayers: [1, 3]
        },
        '13': {
            title: "Cosmological Constants",
            purpose: "To define the fundamental physical parameters of the universe that permit the existence of life and consciousness.",
            components: [
                "Fine-Structure Constant",
                "Gravitational Constant (G)",
                "Planck's Constant (h)"
            ],
            connections: "This layer underpins all other physical layers, particularly Layer 1. It is the bridge to the meta-reality of Layer 14.",
            connectedLayers: [1, 14]
        },
        '14': {
            title: "The Meta-Reality Construct",
            purpose: "The source and destination of the SCPN. A higher-dimensional reality where consciousness is not an emergent property but the fundamental substrate.",
            components: [
                "Universal Branes",
                "Vesica Piscis Bridge",
                "Calabi-Yau Inner Structure",
                "Trans-Brane Network"
            ],
            connections: "Connects to the entire framework through Layer 13, defining the 'rules' of our universe. It is the ultimate context for existence.",
            connectedLayers: [13]
        },
    };

    const scpnDetailData = {
        '1': [
            { id: '1_1', title: 'Logical Coherence', content: 'Detailed analysis of the logical framework supporting quantum effects in biology...'},
            { id: '1_2', title: 'Mathematical Formalism', content: 'The equations governing microtubule quantum states. Includes the Hameroff-Penrose model. <button class="explore-formulation-btn" data-formulation-id="F1_2">Explore Formulation ⇲</button>'},
            { id: '1_3', title: 'Empirical Evidence', content: 'Review of experiments hinting at quantum coherence in biological systems...'}
            // ... more audit points
        ],
    };

    const scpnFormulationData = {
        'F1_2': {
            equation: 'E = ħ / t_G',
            title: "Orch-OR Objective Reduction",
            terms: [
                { term: 'E', description: 'Gravitational self-energy of the microtubule.' },
                { term: 'ħ', description: 'Reduced Planck constant.' },
                { term: 't_G', description: 'Time until objective reduction.' }
            ],
            parameters: [
                { param: 'Superposition Separation', value: 'a' },
            ],
            integration: "This formula from the Hameroff-Penrose Orch-OR theory posits that consciousness arises from a sequence of discrete events, each a moment of objective reduction of a quantum state within the brain's microtubules."
        }
    };

    // --- DOM ELEMENTS ---
    const vizPanel = document.getElementById('visualization-panel');
    const metatronCube = document.getElementById('metatron-cube');
    const metaRealitySceneContainer = document.getElementById('meta-reality-scene');
    const contentPanel = document.getElementById('content-panel');
    const defaultContent = document.getElementById('default-content');
    const layerContent = document.getElementById('layer-content');
    const nodes = document.querySelectorAll('.node');
    const connectionsGroup = document.getElementById('connections');
    
    // --- 3D SCENE (THREE.JS) SETUP ---
    let scene, camera, renderer, controls, bridge;

    function init3DScene() {
        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, vizPanel.clientWidth / vizPanel.clientHeight, 0.1, 1000);
        renderer = new THREE.WebGLRenderer({ antialias: true });
        
        renderer.setSize(vizPanel.clientWidth, vizPanel.clientHeight);
        metaRealitySceneContainer.appendChild(renderer.domElement);

        // Controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        // The Bulk (background)
        scene.background = new THREE.Color(0x000011);

        // The Branes
        const braneGeometry = new THREE.SphereGeometry(15, 32, 32);
        const braneMaterial = new THREE.MeshBasicMaterial({ color: 0x00aaff, transparent: true, opacity: 0.2, wireframe: true });
        const brane1 = new THREE.Mesh(braneGeometry, braneMaterial);
        const brane2 = new THREE.Mesh(braneGeometry, braneMaterial);
        brane1.position.x = -20;
        brane2.position.x = 20;
        scene.add(brane1);
        scene.add(brane2);

        // The Bridge (Vesica Piscis)
        const bridgeShape = new THREE.Shape();
        bridgeShape.moveTo(0, 10);
        bridgeShape.absarc(5, 0, 10, Math.PI * 0.666, Math.PI * 1.333, false);
        bridgeShape.absarc(-5, 0, 10, Math.PI * -0.333, Math.PI * 0.333, false);
        const extrudeSettings = { depth: 2, bevelEnabled: true, bevelSegments: 2, steps: 2, bevelSize: 1, bevelThickness: 1 };
        const bridgeGeometry = new THREE.ExtrudeGeometry(bridgeShape, extrudeSettings);
        const bridgeMaterial = new THREE.MeshStandardMaterial({ color: 0xffaa00, emissive: 0xffaa00, emissiveIntensity: 0.5 });
        bridge = new THREE.Mesh(bridgeGeometry, bridgeMaterial);
        bridge.rotation.x = Math.PI / 2;
        bridge.position.set(-5, 0, 0);
        scene.add(bridge);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const pointLight = new THREE.PointLight(0xffffff, 1);
        pointLight.position.set(0, 0, 10);
        scene.add(pointLight);

        camera.position.z = 50;
        
        animate3D();
    }
    
    function animate3D() {
        if (!metaRealitySceneContainer.classList.contains('hidden')) {
            requestAnimationFrame(animate3D);
            if (bridge) bridge.rotation.y += 0.005;
            controls.update();
            renderer.render(scene, camera);
        }
    }
    
    window.addEventListener('resize', () => {
        if (!metaRealitySceneContainer.classList.contains('hidden')) {
            camera.aspect = vizPanel.clientWidth / vizPanel.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(vizPanel.clientWidth, vizPanel.clientHeight);
        }
    });


    // --- UI LOGIC ---

    function updateContentPanel(layerId) {
        const data = scpnData[layerId];
        if (!data) return;

        defaultContent.classList.add('hidden');
        layerContent.classList.remove('hidden');

        document.getElementById('layer-title').textContent = `Layer ${layerId}: ${data.title}`;
        document.getElementById('layer-purpose').textContent = data.purpose;
        document.getElementById('layer-connections').textContent = data.connections;

        const componentsList = document.getElementById('layer-components');
        componentsList.innerHTML = '';
        data.components.forEach(comp => {
            const li = document.createElement('li');
            li.textContent = comp;
            componentsList.appendChild(li);
        });
        
        const diveDeeperBtn = document.getElementById('dive-deeper-btn');
        if (scpnDetailData[layerId]) {
            diveDeeperBtn.classList.remove('hidden');
            diveDeeperBtn.onclick = () => openDetailModal(layerId);
        } else {
            diveDeeperBtn.classList.add('hidden');
        }
    }

    function highlightNodeAndConnections(activeNode) {
        nodes.forEach(n => n.classList.remove('active'));
        activeNode.classList.add('active');

        connectionsGroup.innerHTML = '';
        const layerId = activeNode.dataset.id;
        const data = scpnData[layerId];
        if (data && data.connectedLayers) {
            const startX = activeNode.getAttribute('cx');
            const startY = activeNode.getAttribute('cy');

            data.connectedLayers.forEach(connId => {
                const endNode = document.querySelector(`.node[data-id='${connId}']`);
                if (endNode) {
                    const endX = endNode.getAttribute('cx');
                    const endY = endNode.getAttribute('cy');
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', startX);
                    line.setAttribute('y1', startY);
                    line.setAttribute('x2', endX);
                    line.setAttribute('y2', endY);
                    line.classList.add('highlight');
                    connectionsGroup.appendChild(line);
                }
            });
        }
    }
    
    function switchTo3DView() {
        metatronCube.classList.add('hidden');
        metaRealitySceneContainer.classList.remove('hidden');
        if (!scene) {
            init3DScene();
        } else {
            // Restart animation loop if returning to this view
            animate3D();
        }
    }

    function switchTo2DView() {
        metaRealitySceneContainer.classList.add('hidden');
        metatronCube.classList.remove('hidden');
    }

    nodes.forEach(node => {
        node.addEventListener('click', (e) => {
            const layerId = e.target.dataset.id;

            if (layerId === '14') {
                switchTo3DView();
            } else {
                switchTo2DView();
            }

            updateContentPanel(layerId);
            highlightNodeAndConnections(e.target);
        });
    });

    // --- MODAL LOGIC ---
    const detailModal = document.getElementById('detail-modal');
    const formulationModal = document.getElementById('formulation-modal');
    
    function openDetailModal(layerId) {
        const data = scpnDetailData[layerId];
        if (!data) return;
        
        const body = document.getElementById('detail-modal-body');
        // A simple representation. For a full Metatron's sub-map, an SVG would be generated here.
        body.innerHTML = `
            <div class="sub-map-container">
                <h4>Layer ${layerId} Audit</h4>
                <ul>${data.map(item => `<li data-content-id="${item.id}" class="sub-map-item">${item.title}</li>`).join('')}</ul>
            </div>
            <div class="detail-content-area">
                <p>Select an item from the list to see details.</p>
            </div>
        `;
        
        detailModal.classList.remove('hidden');

        // Add event listeners to the new sub-map items
        body.querySelectorAll('.sub-map-item').forEach(item => {
            item.addEventListener('click', () => {
                const contentId = item.dataset.contentId;
                const detail = data.find(d => d.id === contentId);
                body.querySelector('.detail-content-area').innerHTML = `<h3>${detail.title}</h3><p>${detail.content}</p>`;
            });
        });

        // Add event listeners for formulation buttons inside the modal
        body.addEventListener('click', function(event) {
            if (event.target.matches('.explore-formulation-btn')) {
                const formulationId = event.target.dataset.formulationId;
                openFormulationModal(formulationId);
            }
        });
    }

    function openFormulationModal(formulationId) {
        const data = scpnFormulationData[formulationId];
        if (!data) return;

        const body = document.getElementById('formulation-modal-body');
        body.innerHTML = `
            <h3>${data.title}</h3>
            <p style="font-size: 1.5em; text-align: center; background: #000; padding: 20px; border-radius: 5px;">${data.equation}</p>
            <h4>Component Breakdown</h4>
            <table>
                ${data.terms.map(t => `<tr><td><strong>${t.term}</strong></td><td>${t.description}</td></tr>`).join('')}
            </table>
            <h4>Simulation Parameters</h4>
            <table>
                ${data.parameters.map(p => `<tr><td>${p.param}</td><td>${p.value}</td></tr>`).join('')}
            </table>
            <h4>Conceptual Integration</h4>
            <p>${data.integration}</p>
        `;
        formulationModal.classList.remove('hidden');
    }

    function closeModal(modal) {
        modal.classList.add('hidden');
    }

    document.querySelectorAll('.close-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            closeModal(e.target.closest('.modal-container'));
        });
    });

    window.addEventListener('click', (e) => {
        if (e.target.classList.contains('modal-container')) {
            closeModal(e.target);
        }
    });

});