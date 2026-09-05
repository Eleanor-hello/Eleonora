import React, { useEffect, useRef } from 'react';

interface Star {
  x: number;
  y: number;
  size: number;
  alpha: number;
  baseAlpha: number;
  twinkleSpeed: number;
  twinklePhase: number;
  color: string;
}

interface ShootingStar {
  x: number;
  y: number;
  length: number;
  speed: number;
  angle: number;
  alpha: number;
  active: boolean;
}

const STAR_COLORS = [
  '#FFFFFF',
  '#E0E7FF',
  '#C7D2FE',
  '#A5B4FC',
  '#E9D5FF',
  '#BAE6FD',
  '#FED7AA',
];

export const CosmicBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      initStars();
    };

    window.addEventListener('resize', handleResize);

    // Create stars
    let stars: Star[] = [];
    const count = Math.min(Math.floor((width * height) / 14000), 90);

    const initStars = () => {
      stars = [];
      for (let i = 0; i < count; i++) {
        const baseAlpha = 0.2 + Math.random() * 0.7;
        stars.push({
          x: Math.random() * width,
          y: Math.random() * height,
          size: 0.6 + Math.random() * 1.5,
          alpha: baseAlpha,
          baseAlpha,
          twinkleSpeed: 0.008 + Math.random() * 0.02,
          twinklePhase: Math.random() * Math.PI * 2,
          color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
        });
      }
    };

    initStars();

    // Occasional shooting star
    let shootingStar: ShootingStar | null = null;
    let lastShootTime = Date.now();

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw subtle space stars
      for (const s of stars) {
        s.twinklePhase += s.twinkleSpeed;
        const currentAlpha = Math.max(
          0.1,
          s.baseAlpha + Math.sin(s.twinklePhase) * 0.35
        );

        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fillStyle = s.color;
        ctx.globalAlpha = currentAlpha;
        ctx.fill();

        // Subtle glow for larger stars
        if (s.size > 1.2) {
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.size * 2.2, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.globalAlpha = currentAlpha * 0.2;
          ctx.fill();
        }
      }

      // Check shooting star trigger (every 12-25 seconds)
      const now = Date.now();
      if (!shootingStar && now - lastShootTime > 14000) {
        if (Math.random() < 0.02) {
          shootingStar = {
            x: Math.random() * width * 0.7 + width * 0.1,
            y: Math.random() * (height * 0.4),
            length: 80 + Math.random() * 60,
            speed: 6 + Math.random() * 5,
            angle: Math.PI / 4 + (Math.random() - 0.5) * 0.2,
            alpha: 1,
            active: true,
          };
          lastShootTime = now;
        }
      }

      if (shootingStar && shootingStar.active) {
        ctx.save();
        ctx.globalAlpha = shootingStar.alpha;
        const endX =
          shootingStar.x - Math.cos(shootingStar.angle) * shootingStar.length;
        const endY =
          shootingStar.y - Math.sin(shootingStar.angle) * shootingStar.length;

        const grad = ctx.createLinearGradient(
          shootingStar.x,
          shootingStar.y,
          endX,
          endY
        );
        grad.addColorStop(0, '#FFFFFF');
        grad.addColorStop(0.3, '#C084FC');
        grad.addColorStop(1, 'transparent');

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(shootingStar.x, shootingStar.y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        ctx.restore();

        shootingStar.x += Math.cos(shootingStar.angle) * shootingStar.speed;
        shootingStar.y += Math.sin(shootingStar.angle) * shootingStar.speed;
        shootingStar.alpha -= 0.015;

        if (
          shootingStar.alpha <= 0 ||
          shootingStar.x > width ||
          shootingStar.y > height
        ) {
          shootingStar = null;
        }
      }

      ctx.globalAlpha = 1;
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {/* Deep nebula radial gradient overlays */}
      <div className="absolute -top-40 -right-40 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-[#7C3AED]/15 via-[#6366F1]/10 to-transparent blur-3xl" />
      <div className="absolute -bottom-40 -left-40 w-[550px] h-[550px] rounded-full bg-gradient-to-tr from-[#06B6D4]/12 via-[#3B82F6]/08 to-transparent blur-3xl" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[700px] h-[500px] rounded-full bg-[#4C1D95]/07 blur-[120px]" />
      
      {/* Dynamic Starfield Canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
    </div>
  );
};
