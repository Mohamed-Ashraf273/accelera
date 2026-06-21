export const getUser = () => {
  return JSON.parse(localStorage.getItem("user"));
};

export const getToken = () => {
  return localStorage.getItem("token");
};

export const authHeaders = () => {
  const token = getToken();
  if (!token) return {};
  return {
    Authorization: `Bearer ${token}`,
  };
};

export const logout = () => {
  localStorage.removeItem("user");
  localStorage.removeItem("token");
};
