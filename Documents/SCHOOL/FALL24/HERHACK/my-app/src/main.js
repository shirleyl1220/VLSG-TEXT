import React, { useState } from 'react';
import './SpotTheDifference.css';

const games = [
  {
    id: 1,
    differences: [
      { x: 100, y: 200, radius: 20 },
      { x: 300, y: 400, radius: 20 },
      // Add more differences for this level
    ],
    image1: 'level1_image1.jpg',
    image2: 'level1_image2.jpg',
  },
  {
    id: 2,
    differences: [
      { x: 150, y: 250, radius: 20 },
      { x: 350, y: 450, radius: 20 },
      // Add more differences for this level
    ],
    image1: 'level2_image1.jpg',
    image2: 'level2_image2.jpg',
  },
  // Add more games/levels here
];

function SpotTheDifference() {
  const [currentGame, setCurrentGame] = useState(null);
  const [foundDifferences, setFoundDifferences] = useState([]);
  const [score, setScore] = useState(0);

  const startGame = (game) => {
    setCurrentGame(game);
    setFoundDifferences([]);
    setScore(0);
  };

  const handleImageClick = (event) => {
    if (!currentGame) return;

    const { offsetX, offsetY } = event.nativeEvent;
    const differenceFound = currentGame.differences.find((difference) => {
      const distance = Math.sqrt(
        (difference.x - offsetX) ** 2 + (difference.y - offsetY) ** 2
      );
      return distance < difference.radius;
    });

    if (differenceFound && !foundDifferences.includes(differenceFound)) {
      setFoundDifferences([...foundDifferences, differenceFound]);
      setScore(score + 1);
    }
  };

  const closeGame = () => {
    setCurrentGame(null);
  };

  return (
    <div className="main-container">
      <h1>Spot the Difference Games</h1>
      <div className="buttons-container">
        {games.map((game) => (
          <button key={game.id} onClick={() => startGame(game)}>
            Start Game {game.id}
          </button>
        ))}
      </div>

      {currentGame && (
        <div className="game-overlay">
          <button className="close-button" onClick={closeGame}>Close</button>
          <h2>Game {currentGame.id}</h2>
          <div className="score">Score: {score}</div>
          <div className="images-container">
            <div className="image" onClick={handleImageClick}>
              <img src={currentGame.image1} alt="Original" />
              {foundDifferences.map((difference, index) => (
                <div
                  key={index}
                  className="marker"
                  style={{
                    left: difference.x,
                    top: difference.y,
                    width: difference.radius * 2,
                    height: difference.radius * 2,
                  }}
                ></div>
              ))}
            </div>
            <div className="image">
              <img src={currentGame.image2} alt="Modified" />
            </div>
          </div>
          {foundDifferences.length === currentGame.differences.length && (
            <div className="win-message">Congratulations! You found all the differences!</div>
          )}
        </div>
      )}
    </div>
  );
}

export default SpotTheDifference;
