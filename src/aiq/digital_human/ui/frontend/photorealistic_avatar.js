// Photorealistic Avatar Renderer using advanced WebGL techniques

class PhotorealisticAvatar {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = null;
        this.program = null;
        this.buffers = {};
        this.textures = {};
        this.uniforms = {};
        this.frame = 0;
        this.expressions = {
            neutral: { eyeOpen: 1.0, mouthOpen: 0.0, browHeight: 0.0 },
            happy: { eyeOpen: 0.8, mouthOpen: 0.3, browHeight: 0.1 },
            thinking: { eyeOpen: 0.7, mouthOpen: 0.0, browHeight: -0.1 },
            surprised: { eyeOpen: 1.3, mouthOpen: 0.4, browHeight: 0.2 }
        };
        this.currentExpression = 'neutral';
        this.targetExpression = 'neutral';
        this.expressionBlend = 0;
        
        this.init();
    }
    
    init() {
        this.gl = this.canvas.getContext('webgl2', {
            antialias: true,
            preserveDrawingBuffer: true,
            alpha: true
        });
        
        if (!this.gl) {
            console.error('WebGL2 not supported');
            return;
        }
        
        this.setupShaders();
        this.createHead();
        this.loadTextures();
        this.setupLighting();
        this.startAnimation();
    }
    
    setupShaders() {
        // Vertex shader with advanced deformation
        const vertexShaderSource = `#version 300 es
        in vec3 a_position;
        in vec3 a_normal;
        in vec2 a_texcoord;
        in vec3 a_tangent;
        
        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform mat4 u_model;
        uniform mat3 u_normalMatrix;
        uniform float u_time;
        uniform float u_eyeOpen;
        uniform float u_mouthOpen;
        uniform float u_browHeight;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec2 v_texcoord;
        out vec3 v_tangent;
        out vec3 v_bitangent;
        out float v_occlusion;
        
        // Facial animation functions
        vec3 animateFace(vec3 pos, vec2 uv) {
            vec3 animated = pos;
            
            // Eye animation
            if (uv.y > 0.6 && uv.y < 0.7 && abs(uv.x - 0.3) < 0.05) {
                animated.y *= u_eyeOpen;
            }
            if (uv.y > 0.6 && uv.y < 0.7 && abs(uv.x - 0.7) < 0.05) {
                animated.y *= u_eyeOpen;
            }
            
            // Mouth animation
            if (uv.y < 0.3 && abs(uv.x - 0.5) < 0.15) {
                animated.y -= u_mouthOpen * 0.1 * (1.0 - abs(uv.x - 0.5) / 0.15);
            }
            
            // Eyebrow animation
            if (uv.y > 0.75 && uv.y < 0.85) {
                animated.y += u_browHeight * 0.05;
            }
            
            // Subtle breathing animation
            animated.y += sin(u_time * 2.0) * 0.002;
            
            return animated;
        }
        
        void main() {
            vec3 animatedPos = animateFace(a_position, a_texcoord);
            vec4 worldPos = u_model * vec4(animatedPos, 1.0);
            
            v_position = worldPos.xyz;
            v_normal = u_normalMatrix * a_normal;
            v_texcoord = a_texcoord;
            v_tangent = u_normalMatrix * a_tangent;
            v_bitangent = cross(v_normal, v_tangent);
            
            // Ambient occlusion approximation
            v_occlusion = clamp(dot(a_normal, vec3(0, 1, 0)) * 0.5 + 0.5, 0.0, 1.0);
            
            gl_Position = u_projection * u_view * worldPos;
        }`;
        
        // Fragment shader with PBR and subsurface scattering
        const fragmentShaderSource = `#version 300 es
        precision highp float;
        
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_texcoord;
        in vec3 v_tangent;
        in vec3 v_bitangent;
        in float v_occlusion;
        
        uniform sampler2D u_diffuse;
        uniform sampler2D u_normal;
        uniform sampler2D u_roughness;
        uniform sampler2D u_subsurface;
        uniform samplerCube u_environment;
        
        uniform vec3 u_lightPos[3];
        uniform vec3 u_lightColor[3];
        uniform vec3 u_viewPos;
        uniform float u_time;
        
        out vec4 fragColor;
        
        // PBR calculations
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }
        
        float distributionGGX(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            
            float num = a2;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = 3.14159265359 * denom * denom;
            
            return num / denom;
        }
        
        float geometrySchlickGGX(float NdotV, float roughness) {
            float r = (roughness + 1.0);
            float k = (r * r) / 8.0;
            
            float num = NdotV;
            float denom = NdotV * (1.0 - k) + k;
            
            return num / denom;
        }
        
        float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = geometrySchlickGGX(NdotV, roughness);
            float ggx1 = geometrySchlickGGX(NdotL, roughness);
            
            return ggx1 * ggx2;
        }
        
        // Subsurface scattering approximation
        vec3 subsurfaceScattering(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 subsurfaceColor) {
            float thickness = 1.0;
            vec3 scatterDir = normalize(lightDir + normal * 0.3);
            float dotLN = max(0.0, dot(scatterDir, -viewDir));
            float scatter = smoothstep(0.0, 1.0, dotLN);
            scatter = pow(scatter, thickness);
            
            return subsurfaceColor * scatter * 0.5;
        }
        
        void main() {
            // Sample textures
            vec3 albedo = texture(u_diffuse, v_texcoord).rgb;
            vec3 normal = texture(u_normal, v_texcoord).rgb * 2.0 - 1.0;
            float roughness = texture(u_roughness, v_texcoord).r;
            vec3 subsurfaceColor = texture(u_subsurface, v_texcoord).rgb;
            
            // Transform normal to world space
            mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
            normal = normalize(TBN * normal);
            
            // View direction
            vec3 V = normalize(u_viewPos - v_position);
            
            // PBR lighting
            vec3 Lo = vec3(0.0);
            
            for(int i = 0; i < 3; i++) {
                // Light calculations
                vec3 L = normalize(u_lightPos[i] - v_position);
                vec3 H = normalize(V + L);
                float distance = length(u_lightPos[i] - v_position);
                float attenuation = 1.0 / (distance * distance);
                vec3 radiance = u_lightColor[i] * attenuation;
                
                // Cook-Torrance BRDF
                vec3 F0 = vec3(0.04);
                F0 = mix(F0, albedo, 0.0); // Non-metallic skin
                vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);
                
                float NDF = distributionGGX(normal, H, roughness);
                float G = geometrySmith(normal, V, L, roughness);
                
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(normal, V), 0.0) * max(dot(normal, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                
                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                
                float NdotL = max(dot(normal, L), 0.0);
                Lo += (kD * albedo / 3.14159265359 + specular) * radiance * NdotL;
                
                // Add subsurface scattering
                Lo += subsurfaceScattering(L, normal, V, subsurfaceColor) * radiance;
            }
            
            // Ambient lighting with IBL
            vec3 ambient = vec3(0.03) * albedo * v_occlusion;
            
            // Environment reflection
            vec3 R = reflect(-V, normal);
            vec3 envColor = texture(u_environment, R).rgb;
            ambient += envColor * 0.1 * (1.0 - roughness);
            
            vec3 color = ambient + Lo;
            
            // Tone mapping and gamma correction
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));
            
            fragColor = vec4(color, 1.0);
        }`;
        
        // Compile shaders
        const vertexShader = this.compileShader(this.gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = this.compileShader(this.gl.FRAGMENT_SHADER, fragmentShaderSource);
        
        // Create program
        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vertexShader);
        this.gl.attachShader(this.program, fragmentShader);
        this.gl.linkProgram(this.program);
        
        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(this.program));
        }
        
        // Get attribute and uniform locations
        this.setupAttributes();
        this.setupUniforms();
    }
    
    compileShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
            return null;
        }
        
        return shader;
    }
    
    createHead() {
        // Create high-resolution head mesh
        const resolution = 64;
        const vertices = [];
        const normals = [];
        const texcoords = [];
        const tangents = [];
        const indices = [];
        
        // Generate sphere with proper UV mapping
        for (let lat = 0; lat <= resolution; lat++) {
            const theta = lat * Math.PI / resolution;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
            
            for (let lon = 0; lon <= resolution; lon++) {
                const phi = lon * 2 * Math.PI / resolution;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);
                
                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;
                
                vertices.push(x * 0.5, y * 0.5, z * 0.5);
                normals.push(x, y, z);
                
                const u = 1 - (lon / resolution);
                const v = 1 - (lat / resolution);
                texcoords.push(u, v);
                
                // Calculate tangent
                const tangent = [-sinPhi, 0, cosPhi];
                tangents.push(...tangent);
            }
        }
        
        // Generate indices
        for (let lat = 0; lat < resolution; lat++) {
            for (let lon = 0; lon < resolution; lon++) {
                const first = (lat * (resolution + 1)) + lon;
                const second = first + resolution + 1;
                
                indices.push(first, second, first + 1);
                indices.push(second, second + 1, first + 1);
            }
        }
        
        // Create buffers
        this.createBuffer('position', new Float32Array(vertices), 3);
        this.createBuffer('normal', new Float32Array(normals), 3);
        this.createBuffer('texcoord', new Float32Array(texcoords), 2);
        this.createBuffer('tangent', new Float32Array(tangents), 3);
        this.buffers.indices = this.createIndexBuffer(new Uint16Array(indices));
        this.indexCount = indices.length;
    }
    
    createBuffer(name, data, size) {
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
        
        this.buffers[name] = {
            buffer: buffer,
            size: size
        };
    }
    
    createIndexBuffer(data) {
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
        return buffer;
    }
    
    loadTextures() {
        // Create placeholder textures with realistic skin properties
        this.textures.diffuse = this.createTexture([255, 220, 177, 255]); // Skin color
        this.textures.normal = this.createTexture([128, 128, 255, 255]); // Neutral normal
        this.textures.roughness = this.createTexture([180, 180, 180, 255]); // Medium roughness
        this.textures.subsurface = this.createTexture([255, 200, 180, 255]); // SSS color
        
        // Load actual textures (placeholder for demo)
        this.loadTexture('diffuse', 'assets/skin_diffuse.jpg');
        this.loadTexture('normal', 'assets/skin_normal.jpg');
        this.loadTexture('roughness', 'assets/skin_roughness.jpg');
        this.loadTexture('subsurface', 'assets/skin_sss.jpg');
        
        // Create environment cubemap
        this.createEnvironmentMap();
    }
    
    createTexture(color) {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D, 0, this.gl.RGBA,
            1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE,
            new Uint8Array(color)
        );
        return texture;
    }
    
    loadTexture(name, url) {
        const image = new Image();
        image.onload = () => {
            const texture = this.gl.createTexture();
            this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
            this.gl.texImage2D(
                this.gl.TEXTURE_2D, 0, this.gl.RGBA,
                this.gl.RGBA, this.gl.UNSIGNED_BYTE, image
            );
            this.gl.generateMipmap(this.gl.TEXTURE_2D);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR_MIPMAP_LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
            this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
            
            this.textures[name] = texture;
        };
        image.src = url;
    }
    
    createEnvironmentMap() {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, texture);
        
        // Create gradient environment
        const size = 256;
        const faces = [
            this.gl.TEXTURE_CUBE_MAP_POSITIVE_X,
            this.gl.TEXTURE_CUBE_MAP_NEGATIVE_X,
            this.gl.TEXTURE_CUBE_MAP_POSITIVE_Y,
            this.gl.TEXTURE_CUBE_MAP_NEGATIVE_Y,
            this.gl.TEXTURE_CUBE_MAP_POSITIVE_Z,
            this.gl.TEXTURE_CUBE_MAP_NEGATIVE_Z
        ];
        
        faces.forEach(face => {
            const data = new Uint8Array(size * size * 4);
            for (let i = 0; i < size * size; i++) {
                const y = Math.floor(i / size) / size;
                const brightness = Math.floor((1 - y) * 50 + 20);
                data[i * 4] = brightness;
                data[i * 4 + 1] = brightness;
                data[i * 4 + 2] = brightness + 10;
                data[i * 4 + 3] = 255;
            }
            
            this.gl.texImage2D(face, 0, this.gl.RGBA, size, size, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, data);
        });
        
        this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_WRAP_R, this.gl.CLAMP_TO_EDGE);
        
        this.textures.environment = texture;
    }
    
    setupLighting() {
        // Three-point lighting setup
        this.lights = [
            { position: [2, 3, 5], color: [1, 0.95, 0.9], intensity: 1.0 }, // Key light
            { position: [-2, 1, 3], color: [0.8, 0.85, 1], intensity: 0.5 }, // Fill light
            { position: [0, -1, -3], color: [1, 1, 1], intensity: 0.3 } // Rim light
        ];
    }
    
    setupAttributes() {
        const attributes = ['position', 'normal', 'texcoord', 'tangent'];
        
        attributes.forEach(name => {
            const location = this.gl.getAttribLocation(this.program, `a_${name}`);
            if (location >= 0) {
                this.buffers[name].location = location;
            }
        });
    }
    
    setupUniforms() {
        const uniforms = [
            'projection', 'view', 'model', 'normalMatrix',
            'time', 'eyeOpen', 'mouthOpen', 'browHeight',
            'lightPos', 'lightColor', 'viewPos',
            'diffuse', 'normal', 'roughness', 'subsurface', 'environment'
        ];
        
        uniforms.forEach(name => {
            this.uniforms[name] = this.gl.getUniformLocation(this.program, `u_${name}`);
        });
    }
    
    setExpression(expression) {
        if (this.expressions[expression]) {
            this.targetExpression = expression;
        }
    }
    
    updateExpression(deltaTime) {
        const speed = 5.0 * deltaTime;
        this.expressionBlend = Math.min(1.0, this.expressionBlend + speed);
        
        if (this.expressionBlend >= 1.0) {
            this.currentExpression = this.targetExpression;
            this.expressionBlend = 0;
        }
        
        const current = this.expressions[this.currentExpression];
        const target = this.expressions[this.targetExpression];
        
        return {
            eyeOpen: this.lerp(current.eyeOpen, target.eyeOpen, this.expressionBlend),
            mouthOpen: this.lerp(current.mouthOpen, target.mouthOpen, this.expressionBlend),
            browHeight: this.lerp(current.browHeight, target.browHeight, this.expressionBlend)
        };
    }
    
    lerp(a, b, t) {
        return a + (b - a) * t;
    }
    
    render() {
        this.gl.clearColor(0, 0, 0, 0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
        
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.CULL_FACE);
        this.gl.cullFace(this.gl.BACK);
        
        this.gl.useProgram(this.program);
        
        // Update time
        const time = performance.now() * 0.001;
        const deltaTime = time - (this.lastTime || 0);
        this.lastTime = time;
        
        // Update expression
        const expression = this.updateExpression(deltaTime);
        
        // Set uniforms
        this.gl.uniform1f(this.uniforms.time, time);
        this.gl.uniform1f(this.uniforms.eyeOpen, expression.eyeOpen);
        this.gl.uniform1f(this.uniforms.mouthOpen, expression.mouthOpen);
        this.gl.uniform1f(this.uniforms.browHeight, expression.browHeight);
        
        // Set matrices
        const aspect = this.canvas.width / this.canvas.height;
        const projection = this.createProjectionMatrix(45, aspect, 0.1, 100);
        const view = this.createViewMatrix([0, 0, 3], [0, 0, 0], [0, 1, 0]);
        const model = this.createModelMatrix(time);
        const normalMatrix = this.createNormalMatrix(model);
        
        this.gl.uniformMatrix4fv(this.uniforms.projection, false, projection);
        this.gl.uniformMatrix4fv(this.uniforms.view, false, view);
        this.gl.uniformMatrix4fv(this.uniforms.model, false, model);
        this.gl.uniformMatrix3fv(this.uniforms.normalMatrix, false, normalMatrix);
        
        // Set lights
        for (let i = 0; i < 3; i++) {
            const light = this.lights[i];
            this.gl.uniform3fv(this.gl.getUniformLocation(this.program, `u_lightPos[${i}]`), light.position);
            this.gl.uniform3fv(this.gl.getUniformLocation(this.program, `u_lightColor[${i}]`), 
                light.color.map(c => c * light.intensity));
        }
        
        // Set view position
        this.gl.uniform3fv(this.uniforms.viewPos, [0, 0, 3]);
        
        // Bind textures
        this.bindTexture(0, this.textures.diffuse, this.uniforms.diffuse);
        this.bindTexture(1, this.textures.normal, this.uniforms.normal);
        this.bindTexture(2, this.textures.roughness, this.uniforms.roughness);
        this.bindTexture(3, this.textures.subsurface, this.uniforms.subsurface);
        
        // Bind environment map
        this.gl.activeTexture(this.gl.TEXTURE4);
        this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, this.textures.environment);
        this.gl.uniform1i(this.uniforms.environment, 4);
        
        // Bind attributes
        Object.entries(this.buffers).forEach(([name, buffer]) => {
            if (buffer.location !== undefined) {
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer.buffer);
                this.gl.enableVertexAttribArray(buffer.location);
                this.gl.vertexAttribPointer(buffer.location, buffer.size, this.gl.FLOAT, false, 0, 0);
            }
        });
        
        // Draw
        this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, this.buffers.indices);
        this.gl.drawElements(this.gl.TRIANGLES, this.indexCount, this.gl.UNSIGNED_SHORT, 0);
        
        this.frame++;
    }
    
    bindTexture(unit, texture, uniform) {
        this.gl.activeTexture(this.gl.TEXTURE0 + unit);
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        this.gl.uniform1i(uniform, unit);
    }
    
    createProjectionMatrix(fov, aspect, near, far) {
        const f = Math.tan(Math.PI * 0.5 - 0.5 * fov * Math.PI / 180);
        const rangeInv = 1.0 / (near - far);
        
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);
    }
    
    createViewMatrix(eye, center, up) {
        // Simple lookAt implementation
        const z = this.normalize(this.subtract(eye, center));
        const x = this.normalize(this.cross(up, z));
        const y = this.cross(z, x);
        
        return new Float32Array([
            x[0], x[1], x[2], 0,
            y[0], y[1], y[2], 0,
            z[0], z[1], z[2], 0,
            eye[0], eye[1], eye[2], 1
        ]);
    }
    
    createModelMatrix(time) {
        // Subtle head movement
        const rotateY = Math.sin(time * 0.5) * 0.1;
        const rotateX = Math.sin(time * 0.7) * 0.05;
        
        const cosY = Math.cos(rotateY);
        const sinY = Math.sin(rotateY);
        const cosX = Math.cos(rotateX);
        const sinX = Math.sin(rotateX);
        
        return new Float32Array([
            cosY, sinX * sinY, cosX * sinY, 0,
            0, cosX, -sinX, 0,
            -sinY, sinX * cosY, cosX * cosY, 0,
            0, 0, 0, 1
        ]);
    }
    
    createNormalMatrix(modelMatrix) {
        // Extract 3x3 rotation part and transpose
        return new Float32Array([
            modelMatrix[0], modelMatrix[4], modelMatrix[8],
            modelMatrix[1], modelMatrix[5], modelMatrix[9],
            modelMatrix[2], modelMatrix[6], modelMatrix[10]
        ]);
    }
    
    // Vector math utilities
    normalize(v) {
        const length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / length, v[1] / length, v[2] / length];
    }
    
    subtract(a, b) {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }
    
    cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }
    
    startAnimation() {
        const animate = () => {
            requestAnimationFrame(animate);
            this.render();
        };
        animate();
    }
    
    resize(width, height) {
        this.canvas.width = width * window.devicePixelRatio;
        this.canvas.height = height * window.devicePixelRatio;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Export for use in main interface
window.PhotorealisticAvatar = PhotorealisticAvatar;