import { useEffect } from "react";
import axios from "axios";

function Dashboard() {

  useEffect(() => {

    const fetchData = async () => {

      const token = localStorage.getItem("token");

      const res = await axios.get(
        "http://localhost:5000/api/protected",
        {
          headers: {
            Authorization: token
          }
        }
      );

      console.log(res.data);

    };

    fetchData();

  }, []);

  return (
    <div>
      <h1>Dashboard</h1>
    </div>
  );
}

export default Dashboard;