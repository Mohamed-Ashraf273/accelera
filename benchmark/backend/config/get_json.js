async function getJson() {
  const response = await fetch(
    `https://drive.google.com/uc?export=download&id=1Qm4mRPVenvkLdZ3Pbfhwum6aeklveMvB`,
  );
    const jsonFile = await response.json();
  return jsonFile
}

module.exports = getJson;
