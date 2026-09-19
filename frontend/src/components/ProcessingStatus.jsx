import {
  Check,
  LoaderCircle,
  AlertCircle,
} from "lucide-react";


export default function ProcessingStatus({
  steps = [],
}) {

  if (!steps.length) {
    return null;
  }


  const visibleSteps =
    steps.filter(
      (step) =>
        step.status !== "pending"
    );


  const activeStep =
    steps.find(
      (step) =>
        step.status === "active"
    );


  const errorStep =
    steps.find(
      (step) =>
        step.status === "error"
    );


  return (

    <div className="processing-status">

      <div className="processing-status__header">

        <div className="processing-status__status-dot" />

        <div className="processing-status__title">

          {errorStep
            ? "Request failed"
            : activeStep
            ? activeStep.label
            : "Processing your request"}

        </div>

      </div>


      {visibleSteps.length > 0 && (

        <div className="processing-status__steps">

          {visibleSteps.map(
            (step) => (

              <div
                className={`processing-step processing-step--${step.status}`}
                key={step.id}
              >

                <div className="processing-step__indicator">

                  {step.status ===
                    "completed" && (

                    <Check
                      size={14}
                      strokeWidth={3}
                    />

                  )}


                  {step.status ===
                    "active" && (

                    <LoaderCircle
                      size={15}
                      className="processing-spin"
                    />

                  )}


                  {step.status ===
                    "error" && (

                    <AlertCircle
                      size={15}
                    />

                  )}

                </div>


                <span className="processing-step__label">
                  {step.label}
                </span>


                {step.status ===
                  "active" && (

                  <span className="processing-step__dots">
                    <span />
                    <span />
                    <span />
                  </span>

                )}

              </div>

            )
          )}

        </div>

      )}

    </div>
  );
}