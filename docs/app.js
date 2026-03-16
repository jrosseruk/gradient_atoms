// ============================================================
// Gradient Atoms — Atomic-themed interactive features
// ============================================================

// --- Theme management ---
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.setTheme(this.theme);
        const toggle = document.getElementById('theme-toggle');
        if (toggle) toggle.addEventListener('click', () => this.toggleTheme());
    }

    setTheme(theme) {
        this.theme = theme;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        const toggle = document.getElementById('theme-toggle');
        if (toggle) {
            toggle.setAttribute('aria-label',
                theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
        }
    }

    toggleTheme() {
        this.setTheme(this.theme === 'light' ? 'dark' : 'light');
        // Let particle system know
        if (window.particleSystem) window.particleSystem.onThemeChange();
    }
}

// --- Floating particle system (atoms!) ---
class ParticleSystem {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.mouse = { x: -1000, y: -1000 };
        this.resize();
        this.createParticles();
        this.bindEvents();
        this.animate();
    }

    get isDark() {
        return document.documentElement.getAttribute('data-theme') === 'dark';
    }

    get colors() {
        return ['#c850c0', '#ff6b6b', '#f5a623', '#4facfe', '#f093fb'];
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        const count = Math.min(60, Math.floor(window.innerWidth * window.innerHeight / 25000));
        this.particles = [];
        for (let i = 0; i < count; i++) {
            this.particles.push(this.createParticle());
        }
    }

    createParticle() {
        const color = this.colors[Math.floor(Math.random() * this.colors.length)];
        const r = Math.random() * 4 + 1.5;
        return {
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            r: r,
            baseR: r,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            color: color,
            alpha: Math.random() * 0.4 + 0.15,
            baseAlpha: Math.random() * 0.4 + 0.15,
            phase: Math.random() * Math.PI * 2,
            // Some particles have "electron rings"
            hasRing: Math.random() < 0.15,
            ringPhase: Math.random() * Math.PI * 2,
        };
    }

    bindEvents() {
        window.addEventListener('resize', () => {
            this.resize();
            this.createParticles();
        });
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
    }

    onThemeChange() {
        // Particles adapt automatically via isDark
    }

    drawParticle(p, time) {
        const ctx = this.ctx;

        // Gentle pulse
        const pulse = Math.sin(time * 0.002 + p.phase) * 0.3 + 1;
        const r = p.baseR * pulse;

        // Mouse repulsion
        const dx = p.x - this.mouse.x;
        const dy = p.y - this.mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const mouseInfluence = Math.max(0, 1 - dist / 150);

        // Glow grows near mouse
        const glowR = r + mouseInfluence * 8;
        const alpha = p.baseAlpha + mouseInfluence * 0.3;

        // Outer glow
        const gradient = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowR * 3);
        gradient.addColorStop(0, p.color + Math.floor(alpha * 80).toString(16).padStart(2, '0'));
        gradient.addColorStop(1, p.color + '00');
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowR * 3, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.globalAlpha = 1;

        // Electron ring for some particles
        if (p.hasRing) {
            const ringAngle = time * 0.001 + p.ringPhase;
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(ringAngle);
            ctx.scale(1, 0.4);
            ctx.beginPath();
            ctx.arc(0, 0, r * 4, 0, Math.PI * 2);
            ctx.strokeStyle = p.color;
            ctx.globalAlpha = alpha * 0.4;
            ctx.lineWidth = 0.8;
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.restore();

            // Tiny electron on the ring
            const eX = p.x + Math.cos(ringAngle * 3) * r * 4;
            const eY = p.y + Math.sin(ringAngle * 3) * r * 4 * 0.4;
            ctx.beginPath();
            ctx.arc(eX, eY, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = alpha * 0.8;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
    }

    drawConnections() {
        const ctx = this.ctx;
        const maxDist = 120;
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const a = this.particles[i];
                const b = this.particles[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < maxDist) {
                    const alpha = (1 - dist / maxDist) * 0.08;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = this.isDark
                        ? `rgba(240, 147, 251, ${alpha})`
                        : `rgba(200, 80, 192, ${alpha})`;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }
        }
    }

    update() {
        for (const p of this.particles) {
            // Mouse repulsion
            const dx = p.x - this.mouse.x;
            const dy = p.y - this.mouse.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 150 && dist > 0) {
                const force = (150 - dist) / 150 * 0.5;
                p.vx += (dx / dist) * force;
                p.vy += (dy / dist) * force;
            }

            // Friction
            p.vx *= 0.99;
            p.vy *= 0.99;

            p.x += p.vx;
            p.y += p.vy;

            // Wrap around edges
            if (p.x < -20) p.x = this.canvas.width + 20;
            if (p.x > this.canvas.width + 20) p.x = -20;
            if (p.y < -20) p.y = this.canvas.height + 20;
            if (p.y > this.canvas.height + 20) p.y = -20;
        }
    }

    animate() {
        const time = performance.now();
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawConnections();
        for (const p of this.particles) {
            this.drawParticle(p, time);
        }
        this.update();
        requestAnimationFrame(() => this.animate());
    }
}

// --- Scroll-triggered fade-in ---
class ScrollReveal {
    constructor() {
        this.sections = document.querySelectorAll('section:not(#hero)');
        this.dividers = document.querySelectorAll('.section-divider');
        this.init();
    }

    init() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

        this.sections.forEach(s => observer.observe(s));

        // Dividers fade in too
        const dividerObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '0.4';
                    entry.target.style.transition = 'opacity 0.8s ease';
                }
            });
        }, { threshold: 0.5 });

        this.dividers.forEach(d => {
            d.style.opacity = '0';
            dividerObserver.observe(d);
        });
    }
}

// --- Scroll progress bar ---
class ScrollProgress {
    constructor() {
        this.bar = document.getElementById('scroll-progress');
        if (this.bar) {
            window.addEventListener('scroll', () => this.update(), { passive: true });
        }
    }

    update() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        this.bar.style.width = progress + '%';
    }
}

// --- Copy citation to clipboard ---
class ClipboardManager {
    constructor() {
        const bibtex = document.getElementById('bibtex');
        if (bibtex) {
            bibtex.addEventListener('click', () => {
                bibtex.select();
                navigator.clipboard.writeText(bibtex.value).then(() => {
                    bibtex.style.borderColor = '#c850c0';
                    bibtex.style.boxShadow = '0 0 20px rgba(200, 80, 192, 0.3)';
                    setTimeout(() => {
                        bibtex.style.borderColor = '';
                        bibtex.style.boxShadow = '';
                    }, 1200);
                }).catch(() => {});
            });
        }
    }
}

// --- Initialize everything ---
document.addEventListener('DOMContentLoaded', () => {
    new ThemeManager();
    new ScrollProgress();
    new ScrollReveal();
    new ClipboardManager();

    const canvas = document.getElementById('particle-canvas');
    if (canvas) {
        // Check for reduced motion preference
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (!prefersReduced) {
            window.particleSystem = new ParticleSystem(canvas);
        }
    }
});

// Re-typeset MathJax on visibility change
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.MathJax) {
        MathJax.typesetPromise().catch(() => {});
    }
});
